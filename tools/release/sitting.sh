#!/usr/bin/env bash
# The release sitting: every cell of a release's table timed and checked on one binary, in one hold.
#
#   tools/release/sitting.sh --cells FILE --binary tree|portable [--tarball PATH] [--version V] [--tree NAME]
#                            [--out FILE] [--hold-owner NAME] [--keep-going] [--fold off|depths] [--resume VERDICT [--only I,J]] [--dry-run]
#   tools/release/sitting.sh --self-test    (stub depth-serve and checks, no box, no card; check-recipes runs it)
#
# Runs on the box (`just release-sitting`). A cell is a model file served by `bloomery-serve` at the serving defaults
# on a card set: its timed arms are tools/ref/depth-serve.sh's (one fresh server a round and cell, serving every depth of the
# cell one after the other under --fold depths, else one a depth and a round; the headline rows), and on round 1's last server its functional phase is tools/release/pass_checks.py (DEPTH_SERVE_AFTER: after
# that server's timed request, under the lease, so its tapes carry a witness). The sitting is the one place a
# release's numbers and its first-request checks meet, on the binary the release ships.
#
# The cells file. Tab-separated, a header line first (blank lines and `#` lines are skipped):
#
#   file/profile  place  cards  ctx  D  checks  [load_s  pp  tg  fold]
#
#   file/profile  `<profile>` (tools/ref/models/<profile>.sh) or `<profile>=<model path>` (BLOOMERY_REF_MODEL for the cell)
#   place         a | bp | -   (`-`: the placement rule's own pick; bp needs `both`)
#   cards         3090 | a6000 | both   (3090 and a6000 time that card alone, both is BLOOMERY_TIMING_CARDS=a6000+3090)
#   ctx           empty: no --ctx-size, the unset rule's pick (the largest step the machine holds); else DEPTH_SERVE_CTX
#   D             the timed prompt lengths, comma-separated (512,4096): one arm each
#   checks        pass_checks.py's arguments for round 1's functional phase, or `-` for none (a timed-only cell)
#   load_s pp tg  optional: seconds a load takes and the cell's prompt and decode tok/s, for the predicted wall only
#   fold          optional (needs the three before it, which may be empty): off | depths | empty (the sitting's --fold). A cell
#                 whose numbers a shared server would change keeps its depths on servers of their own with `off`
#
# Order. The cells sit family-major: every cell of one file/profile together, `bp` before `a` before `-` inside it,
# otherwise in file order. Round 1 runs them in that order, one depth-serve call a cell with BLOOMERY_AB_ROUNDS=1 and
# the hook; round 2 runs them in the reverse order, each cell's arms rotated by one place (depth-serve's rule for
# round 2), with no hook: no functional request ever precedes a timed one on a server. A red cell ends the sitting
# there (the cells and rounds left are named, not run) because the binary is about to change; --keep-going runs the
# round's other cells anyway, and round 2 then runs the cells whose round 1 was green and names the others not run: a red
# cell costs its own two runs, which a --resume re-runs.
#
# Fold. --fold off (the default) is one fresh server a depth and a round, the form the release's headline rows are read in.
# --fold depths gives depth-serve DEPTH_SERVE_FOLD=depths: a cell's round has one server for all its depths (the first request
# of a server is a fresh server's; each later one ends ` [shared]` in its ROW and the means print a fold audit of shared against
# first), so a cell loads twice, not once a depth and round. It is for iteration sittings: a shared row is a warm server's, and a
# fresh server's first request can read slower than it. The fold is part of the cell (a cell line ends fold=<mode>, and a --resume
# of another fold is refused), and the verdict's header names it. A server's round 2 is always its own: round 1's carries the checks.
# The audit lives over the two rounds, not in a depth-serve call: a call is one round, so the depth that is second on its server
# has no first row in the call's own tables. Round 2 rotates the arms by one place, which makes each of a cell's two depths the
# first request of one round and a shared one of the other; fold_audit pairs them over both rounds, prints the pairs on the
# console after the last round and in the verdict, and names every depth that has none (a round not run, a red run, a [cold] row).
# A folded cell therefore has at most two depths (a third would be first in neither round); a cell of more takes fold=off.
#
# Red. A cell's run is red when its depth-serve call exits other than 0, prints fewer ROW lines than the cell has
# arms, tags a row [cpu-busy] (a co-tenant voided it), or, with checks, has no HOOK line of rc 0 and a SESSION line, or
# prints a `check <name>: FAIL`. A cell whose two rounds print different row labels (the label carries the plan's
# place and context) is red `plan drifted`: the two rounds are not one key. The sitting is green only when every run
# of both rounds ran green.
#
# Binary. `--binary portable --tarball PATH` is the release's: tools/release/build.sh's target/portable build, whose
# sha256 must equal the tarball's `<name>/bin/bloomery-serve` (rc 65 before anything starts; depth-serve checks it
# again through DEPTH_SERVE_TARBALL), and the verdict says stage=portable. `--binary tree` is an iteration's: the
# tree's target/release build, stage=tree, never a release verdict (tools/release/publish.sh looks for portable).
#
# The hold. The box-side hold of --hold-owner (default $BLOOMERY_HOLD_OWNER, else rel), /root/bloomery-<owner>-hold
# (tools/ref/lease-probe.sh's), goes up before the first call and comes down on every exit path, so no other box call
# starts between two cells while the lease is free for a moment. Another owner's hold already up is rc 75, nothing run.
# That hold stops new box.sh commands; it cannot stop a GPU gate whose box.sh command began before it, which takes a card in the
# gap the lease leaves between two cells. So a second hold goes up right after it: tools/gpu-gate.sh's sitting hold (--sitting-hold
# up), naming the cards the cells time, alive as long as this process is. A gate whose cards meet it waits at its next poll
# holding no lock, and `any` keeps off the held card. It comes down before the box hold on every exit path; another sitting's live
# hold is rc 75, nothing run.
#
# The quiet box. The hold stops new box commands, not the ones already running: a compile of another track that began
# before it (cargo, cargo-oxide, rustc, llc, ptxas) runs through the first cell and voids its rows ([cpu-busy]). After
# the hold is up and before anything is written or run, the sitting waits until none of those processes runs (read from
# `ps -eo pid=,stat=,comm=` by command name, a zombie being none; `pgrep -f` would match the shell that asks), polling
# every SITTING_QUIET_POLL seconds for up to SITTING_QUIET_WAIT. It prints `[quiet] waiting: <names>` on the first poll
# and then every sixth (once a minute at the default poll) and `[quiet] box quiet after N s`; past the bound it ends rc
# 75 naming the processes, with no verdict written and no cell run. Before every cell the same check runs with a bound
# of 120 s at most; past it the cell runs anyway, the line says so, and depth-serve's own cpu guard tags and reds the
# rows a compile voided. The first look (not the ones between cells) also waits for any compute process on a card the cells time
# (nvidia-smi's list, `gpu:<name>`) and for any gate that holds the card's lock (`gate-lock:<card>`, also while its binary has no
# compute context yet): a gate that was already running when the sitting began is named and waited for, never signalled.
#
# The baseline. With the box quiet and the cards idle, each card the cells time is read once, its used memory, again every
# SITTING_BASELINE_POLL seconds until two reads agree within 64 MiB (a process that has left nvidia-smi's list may still be
# releasing pages); a card not at rest within SITTING_BASELINE_WAIT seconds ends the sitting rc 75, nothing run. The readings go
# to every depth-serve call as DEPTH_SERVE_MEM_BASELINE: before each load the call waits (`[mem-wait]`, bounded) until each card
# is back within 128 MiB of them, else that cell is red by name. A call that read its own baseline would take the previous cell's
# server, still being released, for the idle card.
#
# Resume. `--resume VERDICT` runs part of an earlier sitting's runs and writes one merged verdict (at --out, which by default is
# the earlier verdict's own place: that verdict and its work directory are kept under a UTC stamp). The cells file, the stage and
# the binary must be the earlier sitting's: another stage or cells file is rc 64, another binary sha256 (the header's, or any run's
# stamp) rc 65, before any hold goes up. Without --only it runs exactly the runs the verdict has no green block for: a red run, and
# the round 2 that never ran because round 1 had a red cell. `--only I,J` (cell numbers of the cells file, 3-7 a range) runs
# every run of the named cells, both rounds, green or not. Every other run keeps its block as the verdict has it, stamp included:
# a red one stays red and a missing one `not run`, so the merged verdict is green only when every run of both rounds is. Inside
# the resume a red run ends it there unless --keep-going, and a red round 1 skips round 2, as in a whole sitting. The headers add
# resumed_from (where the earlier verdict is now), resumed_runs and kept_work (its logs and tapes).
#
# The verdict. Each run block's second line is its stamp, `stamp finished=<UTC> binary_sha256=<sha> sitting=<start> pid=<pid>`:
# which sitting ran it, on which binary. --out (default: beside the tarball as <tarball without .tar.gz>.sitting, or target/release-dist/
# <version>-<tree>-tree.sitting) is `key=value` header lines (stage, result, binary, binary_sha256, tarball,
# tarball_sha256, ...), the cells, then one `== round R cell I rc=N result=...` block a run with its ROW, HOOK, check
# and SESSION lines, then each cell's means over both rounds (depth-serve.sh --summary). Its raw logs, and each
# server's session directory with the tapes, are in <out>.d/. A verdict or directory already there is renamed with
# a UTC stamp, never overwritten. The verdict is written on every exit path; a sitting that did not end reads
# result=red.
#
# Knobs (plain names: no binary reads them, so none is a lever):
#   SITTING_LOAD_EST    seconds a load takes for a cell with no load_s column (unset: that term is unpriced)
#   SITTING_CALL_BOUND  seconds one depth-serve call may take (default: its arms x (BLOOMERY_ARM_BOUND + 120) + 300)
#   SITTING_POLL        seconds between reads of a running call's log, which are echoed as it grows (default 1)
#   SITTING_QUIET_WAIT  whole seconds the box may take to go quiet before the first cell (default 1800), the bound of
#                       the check before each cell is at most 120 of them
#   SITTING_QUIET_POLL  seconds between two looks for a running compile (default 10)
#   SITTING_BASELINE_WAIT  whole seconds a card may take to come to rest for the baseline (default 30)
#   SITTING_BASELINE_POLL  seconds between two reads of a card's used memory for the baseline (default 1)
#   SITTING_STUB_DIR    the self-test's: a directory whose depth-serve.sh and checks.sh stand in for the real two (named on
#                       the first line of the run); leave it unset
# The environment is the caller's beyond what a cell sets: BLOOMERY_LEASE_CARD names the sitting's prediction card
# (required), DEPTH_SERVE_N, _PORT, _AFTER_BOUND and the guards' knobs pass through. BLOOMERY_DRY is refused: --dry-run
# is this script's dry run (no hold, no call, no verdict).
#
# Exit: 0 green; 1 red; 2 a file missing; 64 usage or a cells file that does not parse, or a --resume it cannot merge into; 65 the
# binary is not the tarball's member, or not the one a --resume verdict ran on; 69 a card the cells name is not on the machine or
# gives no memory reading; 75 another owner's hold is up (the box hold, or another sitting's GPU hold), a compile or a program on
# a card of another track still runs after SITTING_QUIET_WAIT seconds, or a card is not at rest for the baseline; 130/143 a signal.
set -uo pipefail
__d=${BASH_SOURCE[0]%/*}
[ "$__d" != "${BASH_SOURCE[0]}" ] || __d=.
ROOT=$(cd "$__d/../.." && pwd)
unset __d
HERE=$ROOT/tools/ref
REAL_DS=$HERE/depth-serve.sh
US=$'\037'

die() {
  echo "sitting.sh: $2" >&2
  exit "$1"
}

# sha256_of <file>: the digest alone (sha256sum on the box, shasum where there is none).
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" 2> /dev/null | cut -d' ' -f1
  else
    shasum -a 256 "$1" 2> /dev/null | cut -d' ' -f1
  fi
}

# member_sha <tarball> <member>: the sha256 of the member's bytes; 1 when the tarball holds none.
member_sha() {
  local tmp out
  tmp=$(mktemp "${TMPDIR:-/tmp}/sitting-member.XXXXXX") || return 1
  if ! tar -xzOf "$1" "$2" > "$tmp" 2> /dev/null || [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    return 1
  fi
  out=$(sha256_of "$tmp")
  rm -f "$tmp"
  [ -n "$out" ] || return 1
  echo "$out"
}

# cells_py <records|text> <cells file> <stage>: the cells file validated (rc 64, each fault named with its line),
# then `records` prints one US-separated line a cell in round 1's order (family-major) —
#   C idx profile path place cards ctx deps checks load_s pp tg fold
# — and `text` the plan and the predicted wall for a person.
cells_py() {
  python3 -I - "$@" << 'PY'
import os
import re
import sys

mode, path, stage = sys.argv[1:4]
default_fold = os.environ.get("SITTING_FOLD", "off")
US = "\037"
HEADER = ["file/profile", "place", "cards", "ctx", "D", "checks"]
OPTIONAL = ["load_s", "pp", "tg", "fold"]
PLACE_RANK = {"bp": 0, "a": 1, "-": 2}
SITTING_ROUNDS = 2
LONG_TOKENS = 5855
CHAT_TOKENS = 128


def fail(line, why):
    print(f"sitting.sh: {path}:{line}: {why}", file=sys.stderr)
    sys.exit(64)


def positive(text, what, line, kind=int):
    try:
        v = kind(text)
    except ValueError:
        fail(line, f"{what} is a positive number, got '{text}'")
    if v <= 0:
        fail(line, f"{what} is a positive number, got '{text}'")
    return v


try:
    raw = open(path, encoding="utf-8").read().split("\n")
except OSError as e:
    print(f"sitting.sh: cannot read the cells file {path}: {e.strerror}", file=sys.stderr)
    sys.exit(2)
rows = [(i + 1, ln) for i, ln in enumerate(raw) if ln.strip() and not ln.lstrip().startswith("#")]
if not rows:
    fail(1, "no header line")
hline, htext = rows[0]
head = htext.split("\t")
want = "\t".join(HEADER) + "[\t" + "\t".join(OPTIONAL) + "]"
if head[: len(HEADER)] != HEADER or head[len(HEADER):] != OPTIONAL[: len(head) - len(HEADER)]:
    fail(hline, f"the header is {want.replace(chr(9), ' <TAB> ')}, got '{htext}'")
ncols = len(head)
cells = []
for line, text in rows[1:]:
    f = text.split("\t")
    if len(f) != ncols:
        fail(line, f"{len(f)} columns, the header has {ncols}")
    prof, place, cards, ctx, deps, checks = f[:6]
    opt = f[6:] + [""] * (len(OPTIONAL) - len(f[6:]))
    m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:=(/\S+))?", prof)
    if not m:
        fail(line, f"file/profile is `<profile>` or `<profile>=</absolute/path>`, got '{prof}'")
    profile, mpath = m.group(1), m.group(2) or ""
    if place not in PLACE_RANK:
        fail(line, f"place is a, bp or -, got '{place}'")
    if cards not in ("3090", "a6000", "both"):
        fail(line, f"cards is 3090, a6000 or both, got '{cards}'")
    if place == "bp" and cards != "both":
        fail(line, f"place bp needs cards both (it splits the model over the two cards), got cards '{cards}'")
    if ctx:
        positive(ctx, "ctx (empty: unset)", line)
    dl = deps.split(",")
    for d in dl:
        positive(d, "each D", line)
    if len(set(dl)) != len(dl):
        fail(line, f"D names a depth twice: '{deps}'")
    if not checks.strip() or "\n" in checks:
        fail(line, "checks is pass_checks.py's arguments or `-`")
    load_s, pp, tg, fold = opt
    if fold not in ("", "off", "depths"):
        fail(line, f"fold is off or depths (empty: the sitting's --fold), got '{fold}'")
    # A folded depth is first on its server only in a round the rotation gives it (round r puts depth (r - 1) mod n first),
    # so the audit's first-request row of every depth needs a round each.
    if (fold or default_fold) == "depths" and len(dl) > SITTING_ROUNDS:
        fail(line, f"fold=depths with {len(dl)} depths ({deps}): the sitting runs {SITTING_ROUNDS} rounds, so only {SITTING_ROUNDS} depths "
                   f"are ever the first request of a folded server and the fold audit has no first row for the rest; "
                   f"give this cell fold=off or {SITTING_ROUNDS} depths")
    if load_s:
        positive(load_s, "load_s", line)
    if pp:
        positive(pp, "pp", line, float)
    if tg:
        positive(tg, "tg", line, float)
    cells.append(dict(idx=len(cells) + 1, line=line, prof=prof, profile=profile, path=mpath, place=place,
                      cards=cards, ctx=ctx, deps=deps, checks=checks, load_s=load_s, pp=pp, tg=tg,
                      fold=fold or default_fold))
if not cells:
    fail(hline, "no cell: the header is the only line")

first = {}
for c in cells:
    first.setdefault(c["prof"], len(first))
order = sorted(cells, key=lambda c: (first[c["prof"]], PLACE_RANK[c["place"]], c["idx"]))

if mode == "records":
    for c in order:
        print(US.join(["C", str(c["idx"]), c["profile"], c["path"], c["place"], c["cards"], c["ctx"], c["deps"],
                       c["checks"], c["load_s"], c["pp"], c["tg"], c["fold"]]))
    sys.exit(0)

overhead = int(os.environ["SITTING_OVERHEAD_S"])
n_predict = int(os.environ.get("DEPTH_SERVE_N") or 128)
est = os.environ.get("SITTING_LOAD_EST", "")
total = 0.0
unpriced = []


def show(c, tag):
    return (f"{tag} cell {c['idx']}: {c['prof']} place={c['place']} cards={c['cards']} ctx={c['ctx'] or 'unset'} "
            f"D={c['deps']} checks={c['checks']} fold={c['fold']}")


def servers_of(c, fold):
    """Servers a cell loads over both rounds: one a depth and round, or (fold depths) one a round."""
    return 2 if fold == "depths" else 2 * len(c["deps"].split(","))


print(f"[plan] sitting: stage={stage} cells={len(cells)} file={path}")
print("[plan] round 1 (family-major; each cell one depth-serve call, the hook on its last server):")
for k, c in enumerate(order, 1):
    print("[plan]   " + show(c, f"{k}."))
print("[plan] round 2 (the reverse order; each cell's arms rotated by one; no hook):")
for k, c in enumerate(reversed(order), 1):
    print("[plan]   " + show(c, f"{k}."))
loads = sum(servers_of(c, c["fold"]) for c in cells)
loads_off = sum(servers_of(c, "off") for c in cells)
print(f"[plan] loads: {loads} servers (a cell with fold=depths loads one a round for all its depths; {loads_off} with every cell's fold off)")
for c in cells:
    deps = [int(d) for d in c["deps"].split(",")]
    servers = servers_of(c, c["fold"])
    total += servers * overhead
    load = c["load_s"] or est
    if load:
        total += servers * int(load)
    else:
        unpriced.append(f"cell {c['idx']} load (no load_s, SITTING_LOAD_EST unset)")
    if c["pp"] and c["tg"]:
        pp, tg = float(c["pp"]), float(c["tg"])
        total += 2 * sum(d / pp + (n_predict - 1) / tg for d in deps)
        if c["checks"] != "-":
            total += LONG_TOKENS / pp + 3 * CHAT_TOKENS / tg
    else:
        unpriced.append(f"cell {c['idx']} requests" + (" and checks" if c["checks"] != "-" else "") + " (no pp and tg)")
bound = ">=" if unpriced else "="
print(f"[plan] predicted wall {bound} {total:.0f} s (~{total / 60:.0f} min) [derived per cell: {loads} servers x "
      f"(load + {overhead} s overhead) + 2 x sum(D/pp + (N-1)/tg), N={n_predict}"
      f" + the checks (long prompt {LONG_TOKENS} tokens/pp + 3 x {CHAT_TOKENS}/tg; a toktape client's own prefill probe a request is not priced);"
      f" the load, pp and tg from the cells file]")
if unpriced:
    print("[plan] unpriced (a lower bound): " + "; ".join(unpriced))
PY
}

usage() {
  echo "usage: tools/release/sitting.sh --cells FILE --binary tree|portable [--tarball PATH] [--version V] [--tree NAME] [--out FILE] [--hold-owner NAME] [--keep-going] [--fold off|depths] [--resume VERDICT [--only I,J]] [--dry-run] | --self-test" >&2
}

# ---- the run ---------------------------------------------------------------------------------------------------------

STAGE='' CELLS='' TARBALL='' VERSION='' TREE='' OUT='' OWNER='' KEEP=0 DRY=0 FOLD=off
WORK='' HOLD_FILE='' FINISHED='' RESULT=red REASON='the sitting ended before its last run' STARTED=''
BIN='' BIN_SHA='' BIN_VERSION='' TB_SHA='' TB_MEMBER='' TB_MEMBER_SHA='' GPU_3090='' CARDS_ERROR='' DS='' CHECKS_RUN='' CHECKS_PY=''
CELLREC=() REDS=() NOTRUN=()
RESUME='' ONLY='' ONLYSET='' RESUME_DIR='' RESUME_NOW='' RESUME_PLAN='' KEPT_WORK='' SEL1=() SEL2=()
SIT_CARDS='' BASE_UUIDS='' QUIET_UUIDS='' MEMBASE='' GPU_HOLD_ON='' GPU_GATE=() GPU_A6000=''

# emit_new <file>: the bytes of <file> past EMIT_OFF to stdout, so a call's log shows as it grows.
EMIT_OFF=0
emit_new() {
  local n
  n=$(wc -c < "$1" 2> /dev/null | tr -d ' ')
  { [ -n "$n" ] && [ "$n" -gt "$EMIT_OFF" ]; } || return 0
  tail -c +$((EMIT_OFF + 1)) "$1" 2> /dev/null | head -c $((n - EMIT_OFF))
  EMIT_OFF=$n
}

hold_path() {
  local owner=$1
  echo "${LEASE_HOLDS%%\**}$owner${LEASE_HOLDS#*\*}"
}

hold_up() {
  local others start=''
  others=$(lease_holds_up "$OWNER") || exit $?
  if [ -n "$others" ]; then
    echo "$others" | sed 's/^/sitting.sh: hold up: /' >&2
    die 75 "another owner's hold is up (above): contention, not a failure; nothing ran"
  fi
  HOLD_FILE=$(hold_path "$OWNER")
  if [ -r "/proc/$$/stat" ]; then
    start=$(sed 's/.*) //' "/proc/$$/stat" | awk '{ print $20 }')
  fi
  if ! (
    set -C
    {
      [ -z "$start" ] || echo "pid=$$ start=$start"
      echo "release sitting $STAGE ${VERSION:-$TREE}"
    } > "$HOLD_FILE"
  ) 2> /dev/null; then
    HOLD_FILE=''
    die 75 "the hold of $OWNER is already up ($(hold_path "$OWNER")): another sitting of that owner runs; nothing ran"
  fi
  # A hold that went up before ours, in the instant between the look and the file: this sitting gives way.
  others=$(lease_holds_up "$OWNER") || true
  if [ -n "$others" ] && __lease_gives_way "$OWNER" "$others"; then
    rm -f "$HOLD_FILE"
    HOLD_FILE=''
    die 75 "another hold went up before this one: this sitting gives way; nothing ran"
  fi
  echo "[hold] $(now) up: $HOLD_FILE"
}

hold_down() {
  [ -z "$HOLD_FILE" ] || rm -f "$HOLD_FILE"
  [ -z "$HOLD_FILE" ] || echo "[hold] $(now) down: $HOLD_FILE"
  HOLD_FILE=''
}

# quiet_procs: one `name(pid)` line for each compile process running now (cargo, cargo-oxide, rustc, llc, ptxas; a zombie
# is none), by command name from `ps -eo pid=,stat=,comm=`; nothing when the box is quiet, 1 when ps does not answer (a
# box that cannot be read is not a quiet one). The name is the last path component, as ps prints it on a Mac.
quiet_procs() {
  ps -eo pid=,stat=,comm= | awk '
    $2 ~ /^Z/ { next }
    { n = $3; sub(/.*\//, "", n) }
    n == "cargo" || n == "cargo-oxide" || n == "rustc" || n == "ptxas" || n ~ /^llc(-[0-9]+)?$/ { print n "(" $1 ")" }'
}

# quiet_gpu: one `gpu:<name>(<pid>)` line for each compute process on a card the sitting times (QUIET_UUIDS; nvidia-smi's process
# name is the executable's path, the last component is the name), nothing when those cards run none; 1 when nvidia-smi does not
# answer (a card that cannot be read is not an idle one).
quiet_gpu() {
  local out
  out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits 2> /dev/null) || return 1
  printf '%s\n' "$out" | awk -F', *' -v want=" $QUIET_UUIDS " '
    NF >= 4 && index(want, " " $4 " ") { n = split($2, p, "/"); print "gpu:" p[n] "(" $1 ")" }'
}

# quiet_gates: one `gate-lock:<card>(<pid>)` line for each card lock of a card the cells time that a gate holds (tools/gpu-gate.sh
# holds it for the whole run, also while the binary has no compute context yet, which the compute list cannot show). The probe is a
# shared, non-blocking lock that is let go at once: a gate polls with a non-blocking exclusive try and tries again a second later.
# A lock file nobody ever opened is held by nobody; a lock that cannot be probed is read as held.
quiet_gates() {
  local c f who
  for c in ${SIT_CARDS//,/ }; do
    case $c in
      3090) f=$LOCK_DIR/bloomery-gate.lock ;;
      a6000) f=$LOCK_DIR/bloomery-gate-a6000.lock ;;
      *) continue ;;
    esac
    [ -e "$f" ] || continue
    if ! flock -s -n "$f" true 2> /dev/null; then
      who=$(gate_holder "$f")
      echo "gate-lock:$c($who)"
    fi
  done
}

# gate_holder <lock file>: the pid of a process whose descriptor carries the lock (/proc/<pid>/fdinfo), `?` when none can be found.
gate_holder() {
  python3 -I - "$1" << 'PY'
import os
import sys

try:
    want = os.stat(sys.argv[1])
    names = sorted((n for n in os.listdir("/proc") if n.isdigit()), key=int)
except OSError:
    print("?")
    sys.exit(0)
for name in names:
    d = f"/proc/{name}/fd"
    try:
        fds = os.listdir(d)
    except OSError:
        continue
    for fd in fds:
        try:
            st = os.stat(f"{d}/{fd}")
            info = open(f"/proc/{name}/fdinfo/{fd}").read()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (want.st_dev, want.st_ino) and any(
                ln.startswith("lock:") and "->" not in ln for ln in info.split("\n")):
            print(name)
            sys.exit(0)
print("?")
PY
}

# quiet_names <quiet_procs lines>: the names with their counts, `cargo x2, rustc x24`.
quiet_names() {
  printf '%s\n' "$1" | sed 's/(.*//' | sort | uniq -c | awk '{ printf "%s%s x%s", (NR > 1 ? ", " : ""), $2, $1 }'
}

# quiet_wait initial|between: returns when no compile process runs (and, for `initial`, no compute process on a card the
# sitting times: a gate that was already running when the sitting began, waited for and never signalled). `initial` (after the
# holds are up, before anything is written) ends the sitting rc 75 past SITTING_QUIET_WAIT; `between` (before a cell) waits at
# most 120 s of it and then returns anyway, naming what still runs.
quiet_wait() {
  local kind=$1 limit=${SITTING_QUIET_WAIT:-1800} poll=${SITTING_QUIET_POLL:-10} t0=$SECONDS polls=0 procs gprocs glocks waited
  if [ "$kind" = between ] && [ "$limit" -gt 120 ]; then limit=120; fi
  while :; do
    procs=$(quiet_procs) || die 2 "ps -eo pid=,stat=,comm= does not answer: whether the box is quiet cannot be read"
    if [ "$kind" = initial ]; then
      gprocs=$(quiet_gpu) || die 2 "nvidia-smi --query-compute-apps does not answer: whether the cards the cells time are idle cannot be read"
      glocks=$(quiet_gates)
      procs=$(printf '%s\n%s\n%s\n' "$procs" "$gprocs" "$glocks" | sed '/^$/d')
    fi
    waited=$((SECONDS - t0))
    if [ -z "$procs" ]; then
      if [ "$kind" = initial ] || [ "$polls" != 0 ]; then echo "[quiet] box quiet after $waited s"; fi
      return 0
    fi
    if [ "$waited" -ge "$limit" ]; then
      if [ "$kind" = initial ]; then
        die 75 "processes of other tracks still run after $waited s (SITTING_QUIET_WAIT $limit; compiles, programs on the cards the cells time, gates holding their locks): $(quiet_names "$procs") [$(printf '%s\n' "$procs" | head -n 12 | paste -sd' ' -)]: contention, not a failure; nothing ran"
      fi
      echo "[quiet] between cells: still running after $waited s: $(quiet_names "$procs"): the cell runs anyway and the cpu guard judges its rows"
      return 0
    fi
    if [ $((polls % 6)) = 0 ]; then echo "[quiet] waiting: $(quiet_names "$procs") ($waited s of $limit)"; fi
    polls=$((polls + 1))
    sleep "$poll" 9>&-
  done
}

# cards_needed: SIT_CARDS (the cards the cells time, `3090`, `a6000` or `3090,a6000`, in the form the GPU hold names them) from the
# plan's records. A pure read: no card is touched.
cards_needed() {
  local rec w3090=0 wa=0
  for rec in "${CELLREC[@]}"; do
    cell_fields "$rec"
    case $CF_CARDS in
      3090) w3090=1 ;;
      a6000) wa=1 ;;
      both) w3090=1 wa=1 ;;
    esac
  done
  SIT_CARDS=''
  [ "$w3090" = 0 ] || SIT_CARDS=3090
  [ "$wa" = 0 ] || SIT_CARDS=${SIT_CARDS:+$SIT_CARDS,}a6000
}

# cards_resolve: the UUIDs of SIT_CARDS (tools/ref/cards.sh) into BASE_UUIDS and QUIET_UUIDS, rc 69 for a card a cell times that
# the machine does not have.
cards_resolve() {
  # shellcheck source=tools/ref/cards.sh
  source "$HERE/cards.sh"
  GPU_A6000=${GPU_A6000:-}
  BASE_UUIDS=''
  case ",$SIT_CARDS," in *,3090,*)
    [ -n "$GPU_3090" ] || die 69 "the cells time the 3090 and tools/ref/cards.sh found none: ${CARDS_ERROR:-no reason given}"
    BASE_UUIDS=$GPU_3090 ;;
  esac
  case ",$SIT_CARDS," in *,a6000,*)
    [ -n "$GPU_A6000" ] || die 69 "the cells time the A6000 and tools/ref/cards.sh found none: ${CARDS_ERROR:-no reason given}"
    BASE_UUIDS=${BASE_UUIDS:+$BASE_UUIDS }$GPU_A6000 ;;
  esac
  QUIET_UUIDS=$BASE_UUIDS
}

# card_mem <uuid>: the card's used memory in MiB; empty when nvidia-smi gives no whole number.
card_mem() {
  local v
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2> /dev/null | head -n 1 | tr -d ' ')
  case $v in '' | *[!0-9]*) return 0 ;; esac
  echo "$v"
}

# baseline_read: MEMBASE, `<uuid>=<MiB>[,...]`, each timing card's used memory with the cards idle (quiet_wait's look has just
# found no compute process on them). A card still releasing memory (a process that has left nvidia-smi's list but whose pages are
# still being freed) is no idle baseline: each card is read until two reads SITTING_BASELINE_POLL seconds apart agree within
# 64 MiB, for at most SITTING_BASELINE_WAIT seconds, past which the sitting ends rc 75 naming the card. Every depth-serve call
# of the sitting is held to this baseline (DEPTH_SERVE_MEM_BASELINE), not to what its own start finds.
baseline_read() {
  local uuid prev cur t0 d
  MEMBASE=''
  for uuid in $BASE_UUIDS; do
    t0=$SECONDS prev=''
    while :; do
      cur=$(card_mem "$uuid")
      [ -n "$cur" ] || die 69 "the card $uuid gives no used-memory reading: the sitting's baseline cannot be read"
      if [ -n "$prev" ]; then
        d=$((cur > prev ? cur - prev : prev - cur))
        if [ "$d" -le 64 ]; then break; fi
      fi
      [ $((SECONDS - t0)) -lt "${SITTING_BASELINE_WAIT:-30}" ] ||
        die 75 "the card $uuid's used memory is still changing after $((SECONDS - t0)) s (reads ${prev:-none} and $cur MiB): no idle baseline exists; contention, not a failure; nothing ran"
      prev=$cur
      sleep "${SITTING_BASELINE_POLL:-1}" 9>&-
    done
    MEMBASE=${MEMBASE:+$MEMBASE,}$uuid=$cur
    echo "[baseline] $(now) the card $uuid: $cur MiB used with the cards idle: every call is held to it"
  done
}

# gpu_hold_up / gpu_hold_down: the GPU hold of this sitting (tools/gpu-gate.sh --sitting-hold): the cards the cells time, as long as
# this process lives. It goes up after the box hold, so no gate that began before the box hold takes a card in the gap the lease
# leaves between two cells; it comes down before the box hold does.
gpu_hold_up() {
  local rc=0
  GPU_GATE=(bash "$ROOT/tools/gpu-gate.sh")
  [ -z "${SITTING_STUB_DIR:-}" ] || GPU_GATE+=(--test-locks "$SITTING_STUB_DIR")
  "${GPU_GATE[@]}" --sitting-hold up "$OWNER" "$SIT_CARDS" "$$" || rc=$?
  [ "$rc" = 0 ] || die "$rc" "the GPU hold of the cards the cells time ($SIT_CARDS) did not go up (above): contention if 75; nothing ran"
  GPU_HOLD_ON=1
}

gpu_hold_down() {
  [ -n "$GPU_HOLD_ON" ] || return 0
  GPU_HOLD_ON=''
  "${GPU_GATE[@]}" --sitting-hold down "$OWNER" || true
}

# archive <path>: a verdict or directory from an earlier sitting keeps its content under a UTC stamp (ARCHIVED_TO).
ARCHIVED_TO=''
archive() {
  local to n=0
  ARCHIVED_TO=''
  [ -e "$1" ] || return 0
  to=$1.$(date -u +%Y%m%dT%H%M%SZ)
  while [ -e "$to" ]; do
    n=$((n + 1))
    to=$1.$(date -u +%Y%m%dT%H%M%SZ).$n
  done
  mv "$1" "$to" || die 2 "cannot move the earlier $1 aside"
  ARCHIVED_TO=$to
}

# finish: the EXIT trap. Stops a call still live, writes the verdict, drops the hold.
finish() {
  local rc=$?
  trap - EXIT
  if [ -n "${BG_HANDLES:-}" ]; then bounded_bg_stop > /dev/null 2>&1 || true; fi
  if [ -z "$FINISHED" ] && [ -n "$OUT" ] && [ -n "$WORK" ]; then
    write_verdict || true
  fi
  gpu_hold_down
  hold_down
  [ -z "$RESUME_DIR" ] || rm -rf "$RESUME_DIR"
  exit "$rc"
}

# cell_fields <record>: CF_IDX CF_PROFILE CF_PATH CF_PLACE CF_CARDS CF_CTX CF_DEPS CF_CHECKS (the load, pp and tg columns
# price the predicted wall in the python plan and are not read here).
cell_fields() {
  local tag
  IFS=$US read -r tag CF_IDX CF_PROFILE CF_PATH CF_PLACE CF_CARDS CF_CTX CF_DEPS CF_CHECKS _ _ _ CF_FOLD <<< "$1"
}

# cell_text <record>: one line naming the cell.
cell_text() {
  cell_fields "$1"
  echo "cell $CF_IDX: $CF_PROFILE${CF_PATH:+=$CF_PATH} place=$CF_PLACE cards=$CF_CARDS ctx=${CF_CTX:-unset} D=$CF_DEPS checks=$CF_CHECKS fold=$CF_FOLD"
}

# call_bound <arms>: SITTING_CALL_BOUND, else the call's arms x (arm bound + the memory wait, the plan probe's share
# and the hook's) + 300 s.
call_bound() {
  if [ -n "${SITTING_CALL_BOUND:-}" ]; then
    echo "$SITTING_CALL_BOUND"
  else
    echo $(($1 * (${BLOOMERY_ARM_BOUND:-900} + 120) + 300))
  fi
}

# judge <round> <log> <rc> <arms> <checks>: RUN_RESULT green|red and RUN_WHY of one depth-serve call.
judge() {
  local r=$1 log=$2 rc=$3 arms=$4 checks=$5 rows busy hooks bad sess fails fail
  RUN_RESULT=green RUN_WHY=''
  if [ "$rc" != 0 ]; then
    fail=$(grep -m1 '^FAIL r[0-9]' "$log" 2> /dev/null | cut -c1-240 || true)
    RUN_RESULT=red RUN_WHY="depth-serve.sh exited $rc${fail:+: $fail}"
    return 0
  fi
  rows=$(grep -c '^ROW r[0-9][0-9]* ' "$log" 2> /dev/null || true)
  if [ "${rows:-0}" != "$arms" ]; then
    RUN_RESULT=red RUN_WHY="${rows:-0} ROW lines, the cell has $arms arms"
    return 0
  fi
  busy=$(grep '^ROW r[0-9][0-9]* ' "$log" | grep -c '\[cpu-busy\]' || true)
  if [ "${busy:-0}" != 0 ]; then
    RUN_RESULT=red RUN_WHY="$busy row(s) tagged [cpu-busy]: a co-tenant voided them"
    return 0
  fi
  if [ "$r" = 1 ] && [ "$checks" != - ]; then
    hooks=$(grep -c '^HOOK r[0-9][0-9]* .* rc=' "$log" || true)
    bad=$(grep '^HOOK r[0-9][0-9]* .* rc=' "$log" | grep -vc ' rc=0$' || true)
    sess=$(grep -c '^SESSION rc=0$' "$log" || true)
    fails=$(grep -E '^check [A-Za-z0-9_]+: FAIL' "$log" | sed 's/^check \([A-Za-z0-9_]*\): FAIL.*/\1/' | paste -sd, - || true)
    if [ "${hooks:-0}" = 0 ]; then
      RUN_RESULT=red RUN_WHY="the functional phase did not run (no HOOK line)"
    elif [ "${bad:-0}" != 0 ]; then
      RUN_RESULT=red RUN_WHY="the functional phase's hook exited other than 0"
    elif [ -n "$fails" ]; then
      RUN_RESULT=red RUN_WHY="check FAIL: $fails"
    elif [ "${sess:-0}" = 0 ]; then
      RUN_RESULT=red RUN_WHY="the functional phase printed no 'SESSION rc=0' line"
    fi
  fi
  return 0
}

ROW_LINES='^(ROW r[0-9]+ |HOOK r[0-9]+ |check [A-Za-z0-9_]+: |SESSION rc=|session-plan: |FAIL r[0-9]+ |\[cold\]|\[user-plan\]|\[binary\])'

# run_cell <round> <record> <position> <total>: one depth-serve call for a cell, its block, rows and verdict.
run_cell() {
  local r=$1 rec=$2 pos=$3 total=$4 tree_arm dir log rc=0 n i k arms=() darr=() envv=()
  cell_fields "$rec"
  dir=$WORK/r$r-c$CF_IDX
  mkdir -p "$dir/tmp" || die 2 "cannot make $dir"
  log=$dir/depth-serve.log
  tree_arm=$TREE
  [ "$STAGE" != portable ] || tree_arm=$TREE@portable
  IFS=, read -r -a darr <<< "$CF_DEPS"
  n=${#darr[@]}
  for ((i = 0; i < n; i++)); do
    k=$(((i + r - 1) % n))
    arms+=("$tree_arm:${darr[k]}")
  done
  envv=(-u BLOOMERY_REF_MODEL -u BLOOMERY_REF_MODEL_PROFILE -u DEPTH_SERVE_CTX -u DEPTH_SERVE_PLACE -u DEPTH_SERVE_TARBALL
    -u DEPTH_SERVE_AFTER -u DEPTH_SERVE_MEM_BASELINE -u BLOOMERY_TIMING_GPU -u BLOOMERY_TIMING_CARDS -u BLOOMERY_DRY
    "BLOOMERY_MODEL=$CF_PROFILE" BLOOMERY_AB_ROUNDS=1 "DEPTH_SERVE_MEM_BASELINE=$MEMBASE" "DEPTH_SERVE_FOLD=$CF_FOLD"
    "TMPDIR=$dir/tmp")
  [ -z "$CF_PATH" ] || envv+=("BLOOMERY_REF_MODEL=$CF_PATH")
  [ "$CF_PLACE" = - ] || envv+=("DEPTH_SERVE_PLACE=$CF_PLACE")
  [ -z "$CF_CTX" ] || envv+=("DEPTH_SERVE_CTX=$CF_CTX")
  [ "$STAGE" != portable ] || envv+=("DEPTH_SERVE_TARBALL=$TARBALL")
  case $CF_CARDS in
    both) envv+=(BLOOMERY_TIMING_CARDS=a6000+3090) ;;
    3090) envv+=("BLOOMERY_TIMING_GPU=$GPU_3090") ;;
  esac
  # A card set the user's environment does not see as the timing environment does: the check notes the difference.
  if [ "$CF_CARDS" != both ] && [ -z "${DEPTH_SERVE_USER_PLAN:-}" ]; then envv+=(DEPTH_SERVE_USER_PLAN=note); fi
  if [ "$r" = 1 ] && [ "$CF_CHECKS" != - ]; then
    envv+=("DEPTH_SERVE_AFTER=$CHECKS_RUN $CF_CHECKS")
  fi
  quiet_wait between
  echo
  echo "=== sitting: round $r, $pos of $total: $(cell_text "$rec") ($(now)) ==="
  EMIT_OFF=0
  : > "$log"
  BG_OUT=$log bounded_bg "$(call_bound "$n")" env "${envv[@]}" bash "$DS" "${arms[@]}" || die 70 "cannot start the call"
  while kill -0 "$BG_PID" 2> /dev/null; do
    emit_new "$log" || true
    sleep "${SITTING_POLL:-1}" 9>&-
  done
  bounded_bg_wait
  rc=$?
  emit_new "$log" || true
  judge "$r" "$log" "$rc" "$n" "$CF_CHECKS"
  record_run "$r" "$rec" "$rc" "$log"
  echo "=== sitting: round $r, $pos of $total: cell $CF_IDX $RUN_RESULT${RUN_WHY:+: $RUN_WHY} ==="
  [ "$RUN_RESULT" = green ]
}

# record_run <round> <record> <rc> <log>: the run's verdict block and rows, and the cross-round label check.
record_run() {
  local r=$1 rec=$2 rc=$3 log=$4 block labels other
  cell_fields "$rec"
  block=$WORK/block-r$r-c$CF_IDX.txt
  labels=$WORK/labels-r$r-c$CF_IDX.txt
  grep '^ROW r[0-9][0-9]* ' "$log" | sed 's/^ROW r[0-9]* \([^ ]*\) D=.*/\1/' | sort -u > "$labels" || true
  if [ "$RUN_RESULT" = green ] && [ "$r" = 2 ] && [ -s "$WORK/labels-r1-c$CF_IDX.txt" ]; then
    other=$WORK/labels-r1-c$CF_IDX.txt
    if ! cmp -s "$other" "$labels"; then
      RUN_RESULT=red RUN_WHY="plan drifted: round 1 ran $(paste -sd' ' "$other"), round 2 $(paste -sd' ' "$labels")"
    fi
  fi
  {
    echo "== round $r cell $CF_IDX rc=$rc result=$RUN_RESULT${RUN_WHY:+ reason=$RUN_WHY}"
    echo "stamp finished=$(now) binary_sha256=$BIN_SHA sitting=$STARTED pid=$$"
    grep -E "$ROW_LINES" "$log" | sed "s/^ROW r[0-9][0-9]* /ROW r$r /; s/^FAIL r[0-9][0-9]* /FAIL r$r /" || true
  } > "$block"
  if [ "$RUN_RESULT" = red ]; then
    REDS+=("round $r cell $CF_IDX ($CF_PROFILE place=$CF_PLACE cards=$CF_CARDS): $RUN_WHY")
  fi
}

# write_blocks: the run blocks that exist, round 1 in the plan's order and round 2 in the reverse, whoever ran them.
write_blocks() {
  local r i n=${#CELLREC[@]} rec b
  for r in 1 2; do
    for ((i = 0; i < n; i++)); do
      if [ "$r" = 1 ]; then rec=${CELLREC[i]}; else rec=${CELLREC[n - 1 - i]}; fi
      cell_fields "$rec"
      b=$WORK/block-r$r-c$CF_IDX.txt
      [ -s "$b" ] || continue
      echo
      cat "$b"
    done
  done
}

# fold_cells: the folded cells with their depths, `idx=D,D` words, for fold_audit.
fold_cells() {
  local rec out=''
  for rec in "${CELLREC[@]}"; do
    cell_fields "$rec"
    [ "$CF_FOLD" != depths ] || out+="${out:+ }$CF_IDX=$CF_DEPS"
  done
  echo "$out"
}

# fold_audit: the fold's own term over the whole sitting, which is where a first-request row and a shared one of a depth meet:
# a depth-serve call is one round, so in its own tables the depth that is second on the server has no first row. A row ending
# [shared] followed another request on its server; for each depth of each folded cell with both kinds of warm row ([cold] rows
# out) it is set against the first request's, in percent of the first, and the pairs are pooled (the mean and its 95 % interval
# over pairs: card.py's t table, n - 1 degrees of freedom). One pair has no spread, but a sitting has a pair for every depth of
# every folded cell, and the cells are independent loads. A depth with no pair is named with the rows it has: a round that was not
# run or ended red, a [cold] row (the cold rule's, not the fold's). Prints nothing when no folded cell has a second depth.
fold_audit() {
  local -a cellw
  read -r -a cellw <<< "$(fold_cells)"
  python3 -I - "$HERE/card.py" "$WORK" ${cellw[@]+"${cellw[@]}"} << 'PY'
import glob
import importlib.util
import math
import re
import statistics
import sys

spec = importlib.util.spec_from_file_location('card', sys.argv[1])
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)
ROW = re.compile(r'^ROW r\d+ \S+ D=(\d+) \| tg_tok/s (\S+) \| pp_tok/s (\S+) \| load_s \d+ \| majflt \d+(.*)$')
folded = {}
for word in sys.argv[3:]:
    idx, deps = word.split('=')
    if len(deps.split(',')) > 1:
        folded[int(idx)] = sorted(int(d) for d in deps.split(','))
if not folded:
    sys.exit(0)
rows = {}
for path in sorted(glob.glob(sys.argv[2] + '/block-r*-c*.txt')):
    idx = int(re.search(r'-c(\d+)\.txt$', path).group(1))
    if idx not in folded:
        continue
    for line in open(path, errors='replace'):
        m = ROW.match(line.rstrip('\n'))
        if not m:
            continue
        kind = 'shared' if '[shared]' in m.group(4) else 'first'
        d = rows.setdefault((idx, int(m.group(1))), {'first': [], 'shared': [], 'first_cold': 0, 'shared_cold': 0})
        if '[cold]' in m.group(4):
            d[kind + '_cold'] += 1
        else:
            d[kind].append((float(m.group(2)), float(m.group(3))))
print('== fold audit over the sitting: a [shared] row followed another request on its server; per cell and depth, shared against first, % of first, [cold] rows out')
pairs = []
for idx in sorted(folded):
    for dep in folded[idx]:
        d = rows.get((idx, dep), {'first': [], 'shared': [], 'first_cold': 0, 'shared_cold': 0})
        if not d['first'] or not d['shared']:
            if not (d['first'] or d['shared'] or d['first_cold'] or d['shared_cold']):
                print(f'fold cell {idx} D={dep} | no pair: no row of this depth (no run of the cell printed one)')
            else:
                print(f"fold cell {idx} D={dep} | no pair: first rows {len(d['first'])} warm + {d['first_cold']} [cold], "
                      f"shared rows {len(d['shared'])} warm + {d['shared_cold']} [cold]")
            continue
        pct = [100 * (statistics.mean(x[k] for x in d['shared']) / statistics.mean(x[k] for x in d['first']) - 1) for k in (0, 1)]
        pairs.append(pct)
        print(f'fold cell {idx} D={dep} | tg {pct[0]:+.2f} % | pp {pct[1]:+.2f} %')
if len(pairs) < 2:
    print(f'fold pooled over {len(pairs)} pair(s): no interval')
    sys.exit(0)
t975 = card.t_table()
parts = []
for k, name in enumerate(('tg', 'pp')):
    xs = [p[k] for p in pairs]
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    h = t975(len(xs) - 1) * sd / math.sqrt(len(xs))
    parts.append(f'{name} {m:+.2f} % ± {h:.2f} % (sd {sd:.2f} %) [{"clears 0" if abs(m) > h else "holds 0"}]')
print(f'fold pooled over {len(pairs)} pairs | ' + ' | '.join(parts))
PY
}

# write_verdict: the verdict file from the state so far.
write_verdict() {
  local tmp=$OUT.tmp rec b rows r audit
  FINISHED=1
  {
    echo "sitting-verdict 1"
    echo "stage=$STAGE"
    echo "result=$RESULT"
    [ "$RESULT" = green ] || echo "reason=$REASON"
    echo "version=${VERSION:--}"
    echo "tree=$TREE"
    echo "binary=$BIN"
    echo "binary_sha256=$BIN_SHA"
    echo "binary_version=$BIN_VERSION"
    echo "tarball=${TARBALL:--}"
    echo "tarball_member_sha256=${TB_MEMBER_SHA:--}"
    echo "tarball_sha256=${TB_SHA:--}"
    echo "cells_file=$CELLS"
    echo "lease_card=${BLOOMERY_LEASE_CARD:--}"
    echo "hold_owner=$OWNER"
    echo "work=$WORK"
    echo "keep_going=$KEEP"
    echo "fold=$FOLD"
    echo "started=$STARTED"
    echo "finished=$(now)"
    if [ -n "$RESUME" ]; then
      echo "resumed_from=$RESUME_NOW"
      echo "resumed_runs=$RESUME_PLAN"
      echo "kept_work=$KEPT_WORK"
    fi
    echo
    for rec in "${CELLREC[@]}"; do
      cell_text "$rec"
    done
    write_blocks
    for b in ${NOTRUN[@]+"${NOTRUN[@]}"}; do
      echo
      echo "== $b"
    done
    for rec in "${CELLREC[@]}"; do
      cell_fields "$rec"
      rows=$WORK/rows-c$CF_IDX.txt
      : > "$rows"
      for r in 1 2; do
        [ ! -s "$WORK/block-r$r-c$CF_IDX.txt" ] || grep '^ROW r' "$WORK/block-r$r-c$CF_IDX.txt" >> "$rows" || true
      done
      [ -s "$rows" ] || continue
      echo
      echo "== cell $CF_IDX means over its rows"
      bash "$REAL_DS" --summary "$rows" 2>&1 || echo "(depth-serve.sh --summary exited $?)"
    done
    audit=$(fold_audit 2>&1) || audit="(the fold audit exited $?: $audit)"
    FOLD_AUDIT=$audit
    if [ -n "$audit" ]; then
      echo
      echo "$audit"
    fi
  } > "$tmp" && mv "$tmp" "$OUT" || echo "sitting.sh: cannot write the verdict $OUT" >&2
}

# ---- resume: one merged verdict from an earlier sitting's and this one's runs ---------------------------------------------------

# resume_py <verdict> <dir>: reads a verdict this script wrote. Writes each run block to <dir>/block-r<R>-c<I>.txt and prints,
# US-separated, `H key value` (a header line), `C idx text` (a cell line) and `B round idx result sha` (a run block; sha is its
# stamp's binary_sha256, `-` when it has none). rc 2 for a file it cannot read, 64 for one that is no verdict of this script's.
resume_py() {
  python3 -I - "$1" "$2" << 'PY'
import re
import sys

path, out = sys.argv[1:3]
US = "\037"


def fail(code, why):
    print(f"sitting.sh: --resume {path}: {why}", file=sys.stderr)
    sys.exit(code)


try:
    text = open(path, errors="replace").read()
except OSError as e:
    fail(2, f"cannot read it: {e.strerror}")
paras = [p for p in re.split(r"\n[ \t]*\n", text.strip("\n")) if p.strip()]
if not paras or paras[0].split("\n")[0] != "sitting-verdict 1":
    fail(64, "it is no verdict of this script (its first line is not 'sitting-verdict 1')")
for line in paras[0].split("\n")[1:]:
    if "=" in line:
        k, v = line.split("=", 1)
        print("H", k, v, sep=US)
seen = set()
for para in paras[1:]:
    lines = para.split("\n")
    m = re.match(r"^== round ([12]) cell ([0-9]+) rc=([0-9]+) result=(green|red)", lines[0])
    if m:
        r, i, _, res = m.groups()
        if (r, i) in seen:
            fail(64, f"two blocks for round {r} cell {i}")
        seen.add((r, i))
        with open(f"{out}/block-r{r}-c{i}.txt", "w") as f:
            f.write(para + "\n")
        sha = "-"
        for line in lines[1:]:
            m2 = re.match(r"^stamp .*\bbinary_sha256=([0-9a-f]+)\b", line)
            if m2:
                sha = m2.group(1)
        print("B", r, i, res, sha, sep=US)
        continue
    if all(re.match(r"^cell [0-9]+: ", ln) for ln in lines):
        for ln in lines:
            m3 = re.match(r"^cell ([0-9]+): (.*)$", ln)
            print("C", m3.group(1), m3.group(2), sep=US)
PY
}

# slot_selected <round> <cell idx>: 0 when this sitting runs that slot. With --only, every slot of the named cells (both rounds,
# green or not); without it, every slot the verdict being resumed has no green block for.
slot_selected() {
  local b=$RESUME_DIR/block-r$1-c$2.txt
  if [ -n "$ONLY" ]; then
    case $ONLYSET in *",$2,"*) return 0 ;; esac
    return 1
  fi
  [ -s "$b" ] || return 0
  if head -n 1 "$b" | grep -q ' result=green'; then return 1; fi
  return 0
}

# resume_load: reads $RESUME into $RESUME_DIR and refuses, before any hold goes up, a verdict this sitting cannot merge into:
# another stage or cells file (64), another binary (65: a merged verdict never mixes binaries), a cell number the cells file does
# not have (64), nothing to run (64). SEL1 and SEL2 are the records the sitting runs in each round, in that round's order.
resume_load() {
  local out tag a b c d rec r i sha exp have k lo hi p n total=${#CELLREC[@]} s1='' s2=''
  local -a parts
  RESUME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/sitting-resume.XXXXXX") || die 2 "cannot make a directory for the verdict being resumed"
  out=$(resume_py "$RESUME" "$RESUME_DIR") || exit $?
  RV_STAGE='' RV_SHA='' RV_WORK='' RV_CELLS=() RV_BLOCKS=()
  while IFS=$US read -r tag a b c d; do
    case $tag in
      H) case $a in stage) RV_STAGE=$b ;; binary_sha256) RV_SHA=$b ;; work) RV_WORK=$b ;; esac ;;
      C) RV_CELLS+=("cell $a: $b") ;;
      B) RV_BLOCKS+=("$a $b $c $d") ;;
    esac
  done <<< "$out"
  [ "$RV_STAGE" = "$STAGE" ] || die 64 "--resume $RESUME is a stage=${RV_STAGE:-none} verdict and this sitting is --binary $STAGE: a merged verdict is one stage's"
  { [ -n "$RV_SHA" ] && [ "$RV_SHA" = "$BIN_SHA" ]; } ||
    die 65 "the binary here (sha256 $BIN_SHA) is not the one $RESUME ran on (sha256 ${RV_SHA:-none}): a merged verdict never mixes binaries; run the whole sitting again"
  for have in ${RV_BLOCKS[@]+"${RV_BLOCKS[@]}"}; do
    read -r r i _ sha <<< "$have"
    [ "$sha" = - ] || [ "$sha" = "$BIN_SHA" ] ||
      die 65 "round $r cell $i of $RESUME was run on binary sha256 $sha, not this sitting's $BIN_SHA: a merged verdict never mixes binaries"
  done
  exp=$(for rec in "${CELLREC[@]}"; do cell_text "$rec"; done | sort)
  have=$(printf '%s\n' ${RV_CELLS[@]+"${RV_CELLS[@]}"} | sort)
  [ "$have" = "$exp" ] || die 64 "the cells of $RESUME are not this cells file's: a verdict is resumed with the cells file it was run with, and the --fold it was run with (a cell line ends fold=<mode>)"
  if [ -n "$ONLY" ]; then
    ONLYSET=,
    IFS=, read -r -a parts <<< "$ONLY"
    for p in "${parts[@]}"; do
      case $p in
        *[!0-9-]* | '' | -* | *- | *-*-*) die 64 "--only takes cell numbers, 2,5 or 3-7, got '$p'" ;;
        *-*) lo=${p%-*} hi=${p#*-} ;;
        *) lo=$p hi=$p ;;
      esac
      [ "$lo" -le "$hi" ] || die 64 "--only '$p' counts down"
      for ((k = lo; k <= hi; k++)); do
        have=0
        for rec in "${CELLREC[@]}"; do
          cell_fields "$rec"
          [ "$CF_IDX" != "$k" ] || have=1
        done
        [ "$have" = 1 ] || die 64 "--only names cell $k, and the cells file has $total cells"
        ONLYSET="$ONLYSET$k,"
      done
    done
  fi
  SEL1=() SEL2=()
  for rec in "${CELLREC[@]}"; do
    cell_fields "$rec"
    if slot_selected 1 "$CF_IDX"; then SEL1+=("$rec") s1="$s1${s1:+,}$CF_IDX"; fi
  done
  for ((n = total - 1; n >= 0; n--)); do
    rec=${CELLREC[n]}
    cell_fields "$rec"
    if slot_selected 2 "$CF_IDX"; then SEL2+=("$rec") s2="$s2${s2:+,}$CF_IDX"; fi
  done
  { [ "${#SEL1[@]}" -gt 0 ] || [ "${#SEL2[@]}" -gt 0 ]; } ||
    die 64 "nothing to resume: every run of $RESUME is green (--only I,J re-runs the cells it names)"
  KEPT_WORK=$RV_WORK
  RESUME_PLAN="round 1 cells ${s1:--}; round 2 cells ${s2:--}"
}

# resume_seed: the runs of the verdict being resumed that this sitting does not run are this sitting's, with their stamps: their
# blocks are copied into the work directory as they stand, and a green round 1 keeps its labels for round 2's plan check.
resume_seed() {
  local rec r b
  for rec in "${CELLREC[@]}"; do
    cell_fields "$rec"
    for r in 1 2; do
      b=$RESUME_DIR/block-r$r-c$CF_IDX.txt
      [ -s "$b" ] || continue
      if slot_selected "$r" "$CF_IDX"; then continue; fi
      cp "$b" "$WORK/block-r$r-c$CF_IDX.txt" || die 2 "cannot copy the kept run $b into $WORK"
      if [ "$r" = 1 ] && head -n 1 "$b" | grep -q ' result=green'; then
        grep '^ROW r[0-9][0-9]* ' "$b" | sed 's/^ROW r[0-9]* \([^ ]*\) D=.*/\1/' | sort -u > "$WORK/labels-r1-c$CF_IDX.txt" || true
      fi
    done
  done
}

# merge_state: the reds and the runs not run of the merged verdict, from the blocks as they stand (kept and new): a red block is a
# red; a slot with no block is not run, with the reason the sitting stopped for, else that nothing ran it.
merge_state() {
  local rec r b head why
  REDS=()
  for rec in "${CELLREC[@]}"; do
    cell_fields "$rec"
    for r in 1 2; do
      b=$WORK/block-r$r-c$CF_IDX.txt
      if [ -s "$b" ]; then
        head=$(head -n 1 "$b")
        case $head in
          *' result=red'*)
            why=${head#*' reason='}
            [ "$why" != "$head" ] || why='red'
            REDS+=("round $r cell $CF_IDX ($CF_PROFILE place=$CF_PLACE cards=$CF_CARDS): $why") ;;
        esac
      elif ! printf '%s\n' ${NOTRUN[@]+"${NOTRUN[@]}"} | grep -qF "round $r cell $CF_IDX not run"; then
        NOTRUN+=("round $r cell $CF_IDX not run: ${STOPPED:-no run of it exists in the verdict being resumed, and this sitting did not run it}")
      fi
    done
  done
}

# order_records: CELLREC in round 1's order, from the python plan.
order_records() {
  local recs rec
  recs=$(cells_py records "$CELLS" "$STAGE") || exit $?
  CELLREC=()
  while IFS= read -r rec; do
    [ -z "$rec" ] || CELLREC+=("$rec")
  done <<< "$recs"
  [ "${#CELLREC[@]}" -gt 0 ] || die 64 "no cell in $CELLS"
}

# run_round <round> <record>...: the cells in the order given. A red cell ends the sitting there unless --keep-going;
# what is left is named in NOTRUN. ROUND_RED is 1 when any cell of the round was red.
STOPPED='' ROUND_RED=0 FOLD_AUDIT=''
run_round() {
  local r=$1 rec pos=0 total
  shift
  total=$#
  ROUND_RED=0
  for rec in "$@"; do
    pos=$((pos + 1))
    if [ -n "$STOPPED" ]; then
      cell_fields "$rec"
      NOTRUN+=("round $r cell $CF_IDX not run: $STOPPED")
      continue
    fi
    if ! run_cell "$r" "$rec" "$pos" "$total"; then
      ROUND_RED=1
      if [ "$KEEP" = 0 ]; then STOPPED="the sitting stopped at the red ${REDS[${#REDS[@]} - 1]%%:*}"; fi
    fi
  done
}

# round1_green <cell idx>: 0 when round 1 of the cell has a block and it is green.
round1_green() {
  local b=$WORK/block-r1-c$1.txt
  [ -s "$b" ] && head -n 1 "$b" | grep -q ' result=green'
}

# round2_cells <record>...: R2SEL, the cells round 2 runs, in the order given. Under --keep-going a cell whose round 1 was not green is
# left out and named in NOTRUN: its own two runs are what a --resume re-runs, and the cells whose round 1 was green keep their round 2.
R2SEL=()
round2_cells() {
  local rec skipped=''
  R2SEL=()
  for rec in "$@"; do
    cell_fields "$rec"
    if round1_green "$CF_IDX"; then
      R2SEL+=("$rec")
    else
      NOTRUN+=("round 2 cell $CF_IDX not run: its round 1 was red or not run, so its rows are not one binary's (--resume re-runs it)")
      skipped="$skipped${skipped:+,}$CF_IDX"
    fi
  done
  [ -z "$skipped" ] || echo "[sitting] round 2 leaves out cell(s) $skipped: their round 1 was not green"
}

main() {
  local arg i rec rev=() total

  while [ $# -gt 0 ]; do
    arg=$1
    case $arg in
      --cells | --binary | --tarball | --version | --tree | --out | --hold-owner | --resume | --only)
        [ $# -ge 2 ] || { usage; die 64 "$arg needs a value"; }
        case $arg in
          --cells) CELLS=$2 ;;
          --binary) STAGE=$2 ;;
          --tarball) TARBALL=$2 ;;
          --version) VERSION=$2 ;;
          --tree) TREE=$2 ;;
          --out) OUT=$2 ;;
          --hold-owner) OWNER=$2 ;;
          --resume) RESUME=$2 ;;
          --only) ONLY=$2 ;;
        esac
        shift 2
        ;;
      --keep-going) KEEP=1; shift ;;
      --dry-run) DRY=1; shift ;;
      --fold)
        [ $# -ge 2 ] || { usage; die 64 "--fold needs a value"; }
        case $2 in
          off | depths) FOLD=$2 ;;
          *) usage; die 64 "--fold is off or depths, got '$2'" ;;
        esac
        shift 2
        ;;
      *) usage; die 64 "unknown argument '$arg'" ;;
    esac
  done
  [ -n "$CELLS" ] || { usage; die 64 "--cells is required"; }
  case $STAGE in
    tree | portable) ;;
    *) usage; die 64 "--binary is tree (an iteration's verdict) or portable (the release's), got '${STAGE:-nothing}'" ;;
  esac
  if [ "$STAGE" = portable ] && [ -z "$TARBALL" ]; then
    die 64 "--binary portable needs --tarball PATH: the release's tarball (target/release-dist/bloomery-<version>-<platform>.tar.gz) whose bloomery-serve the timed binary must equal"
  fi
  if [ "$STAGE" = tree ] && [ -n "$TARBALL" ]; then
    die 64 "--tarball is the portable stage's: a tree binary is not the tarball's, and the verdict would check nothing against it"
  fi
  [ -z "$TARBALL" ] || case $TARBALL in *.tar.gz) ;; *) die 64 "--tarball '$TARBALL' is no *.tar.gz" ;; esac
  [ -z "$ONLY" ] || [ -n "$RESUME" ] || die 64 "--only names the cells --resume re-runs; alone it resumes nothing"
  [ -z "$RESUME" ] || [ -r "$RESUME" ] || die 2 "--resume $RESUME: no readable verdict there"
  [ -n "$TREE" ] || TREE=${ROOT##*/}
  [[ $TREE =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 64 "--tree '$TREE' is no tree name (a directory under ~/repo)"
  [ -n "$OWNER" ] || OWNER=${BLOOMERY_HOLD_OWNER:-rel}
  [[ $OWNER =~ ^[A-Za-z0-9_]+$ ]] || die 64 "--hold-owner '$OWNER': letters, digits and _"
  [[ ${SITTING_QUIET_WAIT:-1800} =~ ^[0-9]+$ ]] || die 64 "SITTING_QUIET_WAIT is whole seconds, got '$SITTING_QUIET_WAIT'"
  [[ ${SITTING_QUIET_POLL:-10} =~ ^[0-9]+(\.[0-9]+)?$ ]] || die 64 "SITTING_QUIET_POLL is seconds, got '$SITTING_QUIET_POLL'"
  [ -z "${BLOOMERY_DRY:-}" ] || die 64 "BLOOMERY_DRY is set: --dry-run is this script's dry run (no hold, no call, no verdict); each call's own plan is depth-serve.sh's under BLOOMERY_DRY=1"
  # Every path the verdict and the lines below name is absolute, so a caller elsewhere (sit.sh fetches the verdict by the
  # path printed here) reads it as this script does.
  case $CELLS in /*) ;; *) CELLS=$PWD/$CELLS ;; esac
  case $TARBALL in '' | /*) ;; *) TARBALL=$PWD/$TARBALL ;; esac
  if [ -z "$OUT" ]; then
    if [ -n "$TARBALL" ]; then OUT=${TARBALL%.tar.gz}.sitting; else OUT=$ROOT/target/release-dist/${VERSION:+$VERSION-}$TREE-tree.sitting; fi
  fi
  case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
  case $RESUME in '' | /*) ;; *) RESUME=$PWD/$RESUME ;; esac

  export SITTING_FOLD=$FOLD
  SITTING_OVERHEAD_S=$(sed -n 's/^OVERHEAD_S=\([0-9][0-9]*\)$/\1/p' "$REAL_DS" | head -n1)
  [ -n "$SITTING_OVERHEAD_S" ] || die 2 "tools/ref/depth-serve.sh has no OVERHEAD_S=<seconds> line to price a server's overhead from"
  export SITTING_OVERHEAD_S

  order_records
  cells_py text "$CELLS" "$STAGE" || exit $?

  # The binary and the tarball.
  BIN=$HOME/repo/$TREE/target/release/bloomery-serve
  [ "$STAGE" != portable ] || BIN=$HOME/repo/$TREE/target/portable/release/bloomery-serve
  BIN_SHA=$(sha256_of "$BIN")
  TB_MEMBER=''
  if [ -n "$TARBALL" ]; then
    TB_MEMBER=$(basename "${TARBALL%.tar.gz}")/bin/bloomery-serve
    TB_SHA=$(sha256_of "$TARBALL")
    TB_MEMBER_SHA=$(member_sha "$TARBALL" "$TB_MEMBER") || TB_MEMBER_SHA=''
  fi
  cards_needed
  trap finish EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'exit 129' HUP
  if [ "$DRY" = 1 ]; then
    if [ -z "$BIN_SHA" ]; then echo "[dry] binary $BIN: missing (a run refuses, rc 2)"; else echo "[dry] binary $BIN: sha256 $BIN_SHA"; fi
    if [ -n "$TARBALL" ]; then
      if [ -z "$TB_MEMBER_SHA" ]; then
        echo "[dry] tarball $TARBALL: missing or holds no $TB_MEMBER (a run refuses, rc 2)"
      elif [ "$TB_MEMBER_SHA" = "$BIN_SHA" ]; then
        echo "[dry] binary is the tarball's member $TB_MEMBER"
      else
        echo "[dry] binary is not the tarball's member $TB_MEMBER (sha256 $TB_MEMBER_SHA): a run refuses, rc 65"
      fi
    fi
    echo "[dry] the GPU hold would name the cards the cells time ($SIT_CARDS); the baseline of each is read with the cards idle"
    if [ -n "$RESUME" ]; then
      if [ -z "$BIN_SHA" ]; then
        echo "[dry] resume $RESUME: the binary is missing, so its sha256 cannot be compared with the verdict's"
      else
        resume_load
        echo "[dry] resume $RESUME (same binary, same cells): runs $RESUME_PLAN; every other run keeps its block and stamp"
      fi
    fi
    echo "[dry] no hold up, no call, no verdict; the verdict would be $OUT"
    return 0
  fi
  [ -n "$BIN_SHA" ] || die 2 "no binary at $BIN (stage $STAGE: $([ "$STAGE" = portable ] && echo 'just release-build builds it' || echo 'cargo oxide build --release -p bloomery-gpu-gates --bin bloomery-serve builds it'))"
  if [ -n "$TARBALL" ]; then
    [ -r "$TARBALL" ] || die 2 "no tarball at $TARBALL"
    [ -n "$TB_MEMBER_SHA" ] || die 2 "the tarball $TARBALL holds no $TB_MEMBER"
    [ "$BIN_SHA" = "$TB_MEMBER_SHA" ] || die 65 "$BIN (sha256 $BIN_SHA) is not the member $TB_MEMBER of $TARBALL (sha256 $TB_MEMBER_SHA)"
  fi
  BIN_VERSION=$("$BIN" --version 2> /dev/null | head -n1) || true
  [ -n "$BIN_VERSION" ] || die 2 "$BIN --version printed nothing: the binary does not run here"
  [ -n "${BLOOMERY_LEASE_CARD:-}" ] || die 64 "BLOOMERY_LEASE_CARD is unset: name the sitting's prediction card (docs/cards/<slug>.card; BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=...')"
  [ -r "$BLOOMERY_LEASE_CARD" ] || die 2 "the prediction card $BLOOMERY_LEASE_CARD is not readable"

  # The cards the cells time: their UUIDs key the baseline and the card look, and a cell that times the 3090 alone needs its own.
  cards_resolve

  # shellcheck source=tools/ref/lease-probe.sh
  source "$HERE/lease-probe.sh"
  # shellcheck source=tools/gate-bound.sh
  source "$ROOT/tools/gate-bound.sh"
  if [ -n "${SITTING_STUB_DIR:-}" ]; then
    echo "[stub] SITTING_STUB_DIR=$SITTING_STUB_DIR: depth-serve and the checks come from there (the self-test's)"
    DS=$SITTING_STUB_DIR/depth-serve.sh
    CHECKS_RUN="bash $SITTING_STUB_DIR/checks.sh"
  else
    DS=$REAL_DS
    printf -v CHECKS_PY %q "$ROOT/tools/release/pass_checks.py"
    CHECKS_RUN="python3 $CHECKS_PY"
  fi

  [ -z "$RESUME" ] || resume_load
  hold_up
  gpu_hold_up
  quiet_wait initial
  baseline_read
  archive "$OUT"
  OUT_ARCHIVED=$ARCHIVED_TO
  WORK=$OUT.d
  archive "$WORK"
  WORK_ARCHIVED=$ARCHIVED_TO
  mkdir -p "$WORK" "${OUT%/*}" || die 2 "cannot make $WORK"
  STARTED=$(now)
  if [ -n "$RESUME" ]; then
    RESUME_NOW=$RESUME
    if [ ! -e "$RESUME" ] && [ -n "$OUT_ARCHIVED" ]; then RESUME_NOW=$OUT_ARCHIVED; fi
    if [ "$KEPT_WORK" = "$WORK" ] && [ -n "$WORK_ARCHIVED" ]; then KEPT_WORK=$WORK_ARCHIVED; fi
    resume_seed
    echo "[resume] $RESUME_NOW: this sitting runs $RESUME_PLAN; every other run keeps its block and stamp (the earlier sitting's logs and tapes: $KEPT_WORK)"
  fi
  echo "[sitting] stage=$STAGE binary=$BIN sha256=$BIN_SHA ($BIN_VERSION)"
  [ -z "$TARBALL" ] || echo "[sitting] the binary is the member $TB_MEMBER of $TARBALL"

  total=${#CELLREC[@]}
  for ((i = total - 1; i >= 0; i--)); do rev+=("${CELLREC[i]}"); done
  if [ -n "$RESUME" ]; then
    run_round 1 ${SEL1[@]+"${SEL1[@]}"}
  else
    run_round 1 "${CELLREC[@]}"
  fi
  if [ "$ROUND_RED" = 1 ] && [ "$KEEP" = 0 ]; then
    NOTRUN+=("round 2 skipped: round 1 had a red cell, so the binary changes and round 2's rows would not be its")
    echo "[sitting] round 2 skipped: round 1 had a red cell, so the binary changes"
  else
    if [ -n "$RESUME" ]; then round2_cells ${SEL2[@]+"${SEL2[@]}"}; else round2_cells "${rev[@]}"; fi
    run_round 2 ${R2SEL[@]+"${R2SEL[@]}"}
  fi
  [ -z "$RESUME" ] || merge_state

  if [ "${#REDS[@]}" = 0 ] && [ "${#NOTRUN[@]}" = 0 ]; then
    RESULT=green REASON=''
  else
    RESULT=red REASON=$(printf '%s; ' ${REDS[@]+"${REDS[@]}"})
  fi
  write_verdict
  # The audit is the fold's evidence and its pairs span the two rounds, so the console of the sitting carries it as well as the
  # verdict: each depth-serve call above prints only its own round's tables.
  if [ -n "$FOLD_AUDIT" ]; then
    echo
    echo "$FOLD_AUDIT"
  fi
  echo
  echo "sitting: result=$RESULT stage=$STAGE cells=$total reds=${#REDS[@]}"
  for rec in ${REDS[@]+"${REDS[@]}"}; do echo "sitting: red: $rec"; done
  echo "sitting: verdict $OUT"
  echo "sitting: work $WORK"
  [ "$RESULT" = green ]
}

# ---- --self-test -------------------------------------------------------------------------------------------------------
# A stand-in depth-serve.sh and checks.sh (SITTING_STUB_DIR), a mini tree whose tools/ link to this one, a HOME with a
# stub bloomery-serve and a tarball that holds it, and a holds directory of the case's own: no card, no box, no model.
# Each case is a function that prints nothing and leaves its faults in WHY; the driver runs it on the script, then runs
# the case that guards each mutant on a copy of the script with that one line changed and requires it to fail.

st_stubs() {
  cat > "$1/depth-serve.sh" << 'STUB'
#!/usr/bin/env bash
# The self-test's depth-serve.sh: what it was given goes to $STUB_CALLS, and it prints a call's lines (the hook's before
# the last arm's row, one ROW an arm). The profile's name picks a fault: red* fails, slow* waits for $STUB_RELEASE, drift*
# prints another context on its second call, busy* tags a row, cold* tags the first request of a server [cold], short* prints one row,
# nohook* never runs the hook, flaky* is red on its first call only. Under DEPTH_SERVE_FOLD=depths every row after the first ends [shared] and reads 19 tok/s against 20. Each call line names the box hold, the GPU hold (with the cards it names) and the baseline it was given.
set -u
prof=$BLOOMERY_MODEL
n=$(cat "$STUB_COUNT.$prof" 2> /dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$STUB_COUNT.$prof"
hold=down
[ ! -e "$STUB_HOLD" ] || hold=up
gpuhold=down
[ ! -e "$STUB_GPUHOLD" ] || gpuhold=up:$(sed -n 's/.* cards=\([^ ]*\) .*/\1/p' "$STUB_GPUHOLD")
after=unset
[ -z "${DEPTH_SERVE_AFTER:-}" ] || after=set
echo "call profile=$prof ref=${BLOOMERY_REF_MODEL-unset} refprof=${BLOOMERY_REF_MODEL_PROFILE-unset} place=${DEPTH_SERVE_PLACE-unset} ctx=${DEPTH_SERVE_CTX-unset} cards=${BLOOMERY_TIMING_CARDS-unset} gpu=${BLOOMERY_TIMING_GPU-unset} tarball=${DEPTH_SERVE_TARBALL-unset} after=$after rounds=${BLOOMERY_AB_ROUNDS-unset} userplan=${DEPTH_SERVE_USER_PLAN-unset} hold=$hold gpuhold=$gpuhold membase=${DEPTH_SERVE_MEM_BASELINE-unset} fold=${DEPTH_SERVE_FOLD-unset} arms=$*" >> "$STUB_CALLS"
case $prof in
  red*)
    echo "FAIL r1 ${1%%:*} D=${1#*:}: the stub's server did not answer" >&2
    exit 1
    ;;
  flaky*)
    if [ "$n" = 1 ]; then
      echo "FAIL r1 ${1%%:*} D=${1#*:}: the stub's server did not answer the first time" >&2
      exit 1
    fi
    ;;
  slow*)
    i=0
    while [ ! -e "$STUB_RELEASE" ] && [ "$i" -lt 600 ]; do
      sleep 0.1
      i=$((i + 1))
    done
    ;;
esac
ctx=${DEPTH_SERVE_CTX:-16384}
case $prof in drift*) [ "$n" = 1 ] || ctx=8192 ;; esac
fail=0
k=0
for arm in "$@"; do
  k=$((k + 1))
  d=${arm#*:}
  label=${arm%%:*}/place=${DEPTH_SERVE_PLACE:-a}/ctx=$ctx
  tag=
  case $prof in
    busy*) [ "$k" != 1 ] || tag=' [cpu-busy]' ;;
    cold*) [ "$k" != 1 ] || tag=' [cold]' ;;
    short*) [ "$k" = 1 ] || continue ;;
  esac
  if [ "$k" = "$#" ] && [ -n "${DEPTH_SERVE_AFTER:-}" ] && [ "${prof#nohook}" = "$prof" ]; then
    echo "[hook] r1 $label D=$d: DEPTH_SERVE_AFTER after its timed request"
    mkdir -p "$TMPDIR/session-r1-D$d"
    rc=0
    env "BLOOMERY_SESSION_URL=http://127.0.0.1:1" "BLOOMERY_SESSION_PID=$$" "BLOOMERY_SESSION_STDERR=/dev/null" \
      "BLOOMERY_SESSION_DIR=$TMPDIR/session-r1-D$d" bash -c "$DEPTH_SERVE_AFTER" || rc=$?
    echo "HOOK r1 $label D=$d rc=$rc"
    [ "$rc" = 0 ] || fail=1
  fi
  tg=20.00
  if [ "${DEPTH_SERVE_FOLD:-off}" = depths ] && [ "$k" -gt 1 ]; then
    tag="$tag [shared]"
    tg=19.00
  fi
  echo "ROW r1 $label D=$d | tg_tok/s $tg | pp_tok/s 100.00 | load_s 5 | majflt 0$tag"
done
exit "$fail"
STUB
  cat > "$1/checks.sh" << 'STUB'
#!/usr/bin/env bash
# The self-test's pass_checks.py: it keeps a tape in the session directory and prints the lines pass_checks.py does.
mkdir -p "$BLOOMERY_SESSION_DIR" && echo tape > "$BLOOMERY_SESSION_DIR/tape.json"
case "$*" in
  *--stub-red*)
    echo "check pass_short: PASS ok"
    echo "check pass_long: FAIL harness: the stub"
    echo "SESSION rc=1"
    exit 1
    ;;
  *--stub-failline*)
    echo "check pass_short: FAIL the stub"
    echo "SESSION rc=0"
    exit 0
    ;;
  *--stub-nosession*)
    echo "check pass_short: PASS ok"
    exit 0
    ;;
esac
echo "check pass_short: PASS ok"
echo "check pass_long: PASS ok"
echo "SESSION rc=0"
STUB
}

# st_ps <bin dir>: the case's `ps`. `ps -eo pid=,stat=,comm=` (the quiet check's look) answers from $C/ps.seq: one line a
# call, words `name:pid` (`name:pid:Z` a zombie) or `-` for none, the last line again once they run out, no file a quiet
# box, $C/ps.fail a ps that fails; every other call is the real ps's, so the host's own compiles never reach a case.
st_ps() {
  cat > "$1/ps" << 'STUB'
#!/bin/sh
case "$*" in *pid=,stat=,comm=*) ;; *) exec REAL_PS "$@" ;; esac
if [ -e CASE_DIR/ps.fail ]; then
  echo 'ps: the stub fails' >&2
  exit 1
fi
[ -r CASE_DIR/ps.seq ] || exit 0
i=$(cat CASE_DIR/ps.i 2> /dev/null || echo 0)
i=$((i + 1))
echo "$i" > CASE_DIR/ps.i
line=$(sed -n "${i}p" CASE_DIR/ps.seq)
[ -n "$line" ] || line=$(tail -n 1 CASE_DIR/ps.seq)
for w in $line; do
  [ "$w" != - ] || continue
  rest=${w#*:}
  st=S
  case $rest in *:*) st=${rest#*:} ;; esac
  echo "${rest%%:*} $st ${w%%:*}"
done
STUB
  sed -i.bak -e "s|REAL_PS|$(command -v ps)|" -e "s|CASE_DIR|$C|g" "$1/ps" && rm -f "$1/ps.bak"
  chmod +x "$1/ps"
}

# st_nvsmi <bin dir>: the case's `nvidia-smi`. The cards are GPU-aaaa (the A6000) and GPU-bbbb (the 3090). Used memory is 100 MiB, or the
# lines of $C/mem.<uuid>.seq one a look (the last again once they run out); the compute list is the lines of $C/gpu.seq one a look,
# words `pid:name:MiB:uuid` or `-` for none, the last line again once they run out, no file an idle box; $C/gpu.fail makes it fail.
st_nvsmi() {
  cat > "$1/nvidia-smi" << 'STUB'
#!/bin/sh
case "$*" in
  *--query-gpu=name,uuid*) printf 'NVIDIA RTX A6000, GPU-aaaa\nNVIDIA GeForce RTX 3090, GPU-bbbb\n' ;;
  *--query-gpu=memory.used*)
    u=''
    prev=''
    for a in "$@"; do
      [ "$prev" != -i ] || u=$a
      prev=$a
    done
    f=CASE_DIR/mem.$u.seq
    if [ -r "$f" ]; then
      i=$(cat "CASE_DIR/mem.$u.i" 2> /dev/null || echo 0)
      i=$((i + 1))
      echo "$i" > "CASE_DIR/mem.$u.i"
      v=$(sed -n "${i}p" "$f")
      [ -n "$v" ] || v=$(tail -n 1 "$f")
      echo "$v"
    else
      echo 100
    fi ;;
  *--query-compute-apps*)
    if [ -e CASE_DIR/gpu.fail ]; then
      echo 'nvidia-smi: the stub fails' >&2
      exit 9
    fi
    [ -r CASE_DIR/gpu.seq ] || exit 0
    i=$(cat CASE_DIR/gpu.i 2> /dev/null || echo 0)
    i=$((i + 1))
    echo "$i" > CASE_DIR/gpu.i
    line=$(sed -n "${i}p" CASE_DIR/gpu.seq)
    [ -n "$line" ] || line=$(tail -n 1 CASE_DIR/gpu.seq)
    for w in $line; do
      [ "$w" != - ] || continue
      echo "$w" | awk -F: '{ printf "%s, /r/target/release/%s, %s, %s\n", $1, $2, $3, $4 }'
    done ;;
esac
STUB
  sed -i.bak -e "s|CASE_DIR|$C|g" "$1/nvidia-smi" && rm -f "$1/nvidia-smi.bak"
  chmod +x "$1/nvidia-smi"
}

# st_case <name>: a fresh case directory C and TBN (the tarball's name), WHY empty.
st_case() {
  C=$T/$1
  TBN=bloomery-0.0.0-linux-x86_64-cuda-sm86
  mkdir -p "$C/home/repo/stubtree/target/portable/release" "$C/home/repo/stubtree/target/release" "$C/tb/$TBN/bin" \
    "$C/stub" "$C/holds" "$C/bin" "$C/tmp"
  printf '#!/bin/sh\necho "bloomery-serve 0.0.0 (commit stub)"\n' > "$C/home/repo/stubtree/target/portable/release/bloomery-serve"
  chmod +x "$C/home/repo/stubtree/target/portable/release/bloomery-serve"
  cp "$C/home/repo/stubtree/target/portable/release/bloomery-serve" "$C/home/repo/stubtree/target/release/bloomery-serve"
  cp "$C/home/repo/stubtree/target/portable/release/bloomery-serve" "$C/tb/$TBN/bin/bloomery-serve"
  tar -czf "$C/tb/$TBN.tar.gz" -C "$C/tb" "$TBN"
  echo 'kind: baseline' > "$C/card"
  st_stubs "$C/stub"
  st_nvsmi "$C/bin"
  mkdir -p "$C/locks"
  command -v flock > /dev/null || cp "$HERE/card-tests/bin/flock" "$C/bin/flock"
  st_ps "$C/bin"
  if ! command -v timeout > /dev/null; then
    cp "$HERE/stub-timeout.py" "$C/bin/timeout"
    chmod +x "$C/bin/timeout"
  fi
  : > "$C/calls"
  WHY='' ST_EXTRA=()
  mkdir -p "$C/tree/tools/release"
  cp "$SUT" "$C/tree/tools/release/sitting.sh"
  ln -s "$ROOT/tools/ref" "$C/tree/tools/ref"
  ln -s "$ROOT/tools/gate-bound.sh" "$C/tree/tools/gate-bound.sh"
  ln -s "$ROOT/tools/gpu-gate.sh" "$C/tree/tools/gpu-gate.sh"
}

# st_run <args>...: the sitting on the case's cells file, its output in $C/out and its status in RC.
st_run() {
  cd "$C" || exit 2
  env "HOME=$C/home" "PATH=$C/bin:$PATH" "TMPDIR=$C/tmp" "STUB_CALLS=$C/calls" "STUB_COUNT=$C/count" \
    "STUB_HOLD=$C/holds/bloomery-rel-hold" "STUB_RELEASE=$C/release" "BLOOMERY_LEASE_HOLDS=$C/holds/bloomery-*-hold" \
    "BLOOMERY_LEASE_CARD=$C/card" "SITTING_STUB_DIR=$C/stub" "STUB_GPUHOLD=$C/stub/sitting.gpuhold" "BLOOMERY_LOCK_DIR=$C/locks" SITTING_POLL=0.05 SITTING_QUIET_POLL=0.05 SITTING_BASELINE_POLL=0.05 \
    ${ST_EXTRA[@]+"${ST_EXTRA[@]}"} \
    bash "$C/tree/tools/release/sitting.sh" --cells "$C/cells.tsv" --tree stubtree --out "$C/v.sitting" "$@" > "$C/out" 2>&1
  RC=$?
  cd "$OLDPWD" || exit 2
}

# st_start <args>...: the same in the background; its pid in ST_PID and in the cleanup's list.
st_start() {
  cd "$C" || exit 2
  env "HOME=$C/home" "PATH=$C/bin:$PATH" "TMPDIR=$C/tmp" "STUB_CALLS=$C/calls" "STUB_COUNT=$C/count" \
    "STUB_HOLD=$C/holds/bloomery-rel-hold" "STUB_RELEASE=$C/release" "BLOOMERY_LEASE_HOLDS=$C/holds/bloomery-*-hold" \
    "BLOOMERY_LEASE_CARD=$C/card" "SITTING_STUB_DIR=$C/stub" "STUB_GPUHOLD=$C/stub/sitting.gpuhold" "BLOOMERY_LOCK_DIR=$C/locks" SITTING_POLL=0.05 SITTING_QUIET_POLL=0.05 SITTING_BASELINE_POLL=0.05 \
    ${ST_EXTRA[@]+"${ST_EXTRA[@]}"} \
    bash "$C/tree/tools/release/sitting.sh" --cells "$C/cells.tsv" --tree stubtree --out "$C/v.sitting" "$@" > "$C/out" 2>&1 &
  ST_PID=$!
  echo "$ST_PID" >> "$T/pids"
  cd "$OLDPWD" || exit 2
}

st_cells() { tr '|' '\t' > "$C/cells.tsv"; }
HDR='file/profile|place|cards|ctx|D|checks'
portable() { echo --binary portable --tarball "$C/tb/$TBN.tar.gz"; }

fault() { WHY="$WHY; $*"; }
want_rc() { [ "$RC" = "$1" ] || fault "rc $RC, want $1"; }
want_out() { grep -qF -- "$1" "$C/out" || fault "the output lacks '$1'"; }
want_v() { grep -qF -- "$1" "$C/v.sitting" 2> /dev/null || fault "the verdict lacks '$1'"; }
no_calls() { [ ! -s "$C/calls" ] || fault "a call happened"; }
no_hold() { [ ! -e "$C/holds/bloomery-rel-hold" ] || fault "the hold is still up"; }
want_before() {
  local a b
  a=$(grep -nF -- "$1" "$C/out" | head -n 1 | cut -d: -f1)
  b=$(grep -nF -- "$2" "$C/out" | head -n 1 | cut -d: -f1)
  { [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]; } || fault "'$1' (line ${a:-none}) is not before '$2' (line ${b:-none})"
}
# st_fresh: the next sitting of a case starts from no call, no count, no ps position and no verdict.
st_fresh() { : > "$C/calls"; rm -rf "$C"/count.* "$C/ps.i" "$C/v.sitting"; }
calls_seq() {
  local got
  got=$(sed -n "$1" "$C/calls" 2> /dev/null | paste -sd' ' -)
  [ "$got" = "$2" ] || fault "calls '$1' read '$got', want '$2'"
}
CALL_PROF='s/^call profile=\([^ ]*\) .*/\1/p'

case_order() {
  st_case "order$SUF"
  { echo "$HDR"; echo 'pa=/m/a.gguf|a|3090||512,4096|--stub-pass'; echo 'pb=/m/b|bp|both||512|-'; echo 'pa=/m/a.gguf|bp|both||512,4096|--stub-pass'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/^call profile=\([^ ]*\) .* place=\([^ ]*\) .*/\1:\2/p' 'pa:bp pa:a pb:bp pb:bp pa:a pa:bp'
  calls_seq 's/^call .* after=\([a-z]*\) .*/\1/p' 'set set unset unset unset unset'
  calls_seq 's/^call .* rounds=\([^ ]*\) .*/\1/p' '1 1 1 1 1 1'
  calls_seq '1s/.* arms=//p' 'stubtree@portable:512 stubtree@portable:4096'
  calls_seq '6s/.* arms=//p' 'stubtree@portable:4096 stubtree@portable:512'
  [ -z "$WHY" ]
}

case_redstop() {
  st_case "redstop$SUF"
  { echo "$HDR"; echo 'ok1|bp|both||512|--stub-pass'; echo 'red1|bp|both||512|-'; echo 'ok2|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  calls_seq "$CALL_PROF" 'ok1 red1'
  want_out 'sitting: red: round 1 cell 2 (red1 place=bp cards=both): depth-serve.sh exited 1: FAIL r1 stubtree@portable D=512: the stub'"'"'s server did not answer'
  want_out 'round 2 skipped'
  want_v 'result=red'
  want_v 'round 1 cell 3 not run'
  [ -z "$WHY" ]
}

case_keepgoing() {
  st_case "keepgoing$SUF"
  { echo "$HDR"; echo 'ok1|bp|both||512|--stub-pass'; echo 'red1|bp|both||512|-'; echo 'ok2|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --keep-going
  want_rc 1
  calls_seq "$CALL_PROF" 'ok1 red1 ok2 ok2 ok1'
  want_out 'round 2 leaves out cell(s) 2: their round 1 was not green'
  want_v 'round 2 cell 2 not run'
  want_v '== round 2 cell 1 rc=0 result=green'
  want_v '== round 2 cell 3 rc=0 result=green'
  [ -z "$WHY" ]
}

case_verdict() {
  st_case "verdict$SUF"
  { echo "$HDR"; echo 'pa=/m/a.gguf|a|3090||512,4096|--stub-pass'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --version 0.0.0 --fold off
  want_rc 0
  local sha lab=stubtree@portable/place=a/ctx=16384
  sha=$(sha256_of "$C/home/repo/stubtree/target/portable/release/bloomery-serve")
  want_v 'stage=portable'
  want_v 'result=green'
  want_v 'version=0.0.0'
  want_v "binary_sha256=$sha"
  want_v "tarball_member_sha256=$sha"
  grep -q "^stamp finished=.* binary_sha256=$sha sitting=.* pid=[0-9]*\$" "$C/v.sitting" 2> /dev/null || fault "a run block carries no stamp naming the binary"
  want_v "tarball=$C/tb/$TBN.tar.gz"
  want_v 'binary_version=bloomery-serve 0.0.0 (commit stub)'
  want_v 'cell 1: pa=/m/a.gguf place=a cards=3090 ctx=unset D=512,4096 checks=--stub-pass'
  want_v '== round 1 cell 1 rc=0 result=green'
  want_v '== round 2 cell 1 rc=0 result=green'
  want_v "ROW r1 $lab D=512 | tg_tok/s 20.00"
  want_v "ROW r2 $lab D=4096 | tg_tok/s 20.00"
  want_v "HOOK r1 $lab D=4096 rc=0"
  want_v 'check pass_long: PASS ok'
  want_v 'SESSION rc=0'
  want_v '== cell 1 means over its rows'
  want_v "mean $lab D=512 | tg 20.00 tok/s"
  want_out 'sitting: result=green stage=portable cells=1 reds=0'
  find "$C/v.sitting.d" -name tape.json 2> /dev/null | grep -q . || fault "no session tape under the work directory"
  [ -z "$WHY" ]
}

case_tree() {
  st_case "tree$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  st_run --binary tree
  want_rc 0
  want_v 'stage=tree'
  want_v "binary=$C/home/repo/stubtree/target/release/bloomery-serve"
  want_v 'tarball=-'
  calls_seq '1s/.* arms=//p' 'stubtree:512'
  calls_seq 's/.* tarball=\([^ ]*\) .*/\1/p' 'unset unset'
  st_run --binary tree --tarball "$C/tb/$TBN.tar.gz"
  want_rc 64
  want_out "is the portable stage's"
  [ -z "$WHY" ]
}

case_noportable() {
  st_case "noportable$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  st_run --binary portable
  want_rc 64
  want_out '--binary portable needs --tarball'
  no_calls
  no_hold
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written"
  [ -z "$WHY" ]
}

case_mismatch() {
  st_case "mismatch$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  printf '#!/bin/sh\necho "bloomery-serve 9.9.9 (commit other)"\n' > "$C/home/repo/stubtree/target/portable/release/bloomery-serve"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 65
  want_out 'is not the member'
  no_calls
  no_hold
  [ -z "$WHY" ]
}

case_checks() {
  st_case "checks$SUF"
  local flag why prof
  for flag in '--stub-red|depth-serve.sh exited 1|okc' '--stub-failline|check FAIL: pass_short|okc' "--stub-nosession|no 'SESSION rc=0' line|okc" '--stub-pass|no HOOK line|nohook1'; do
    prof=${flag##*|}
    why=${flag#*|}
    why=${why%|*}
    flag=${flag%%|*}
    : > "$C/calls"
    rm -f "$C/count".*
    { echo "$HDR"; echo "$prof|bp|both||512|$flag"; } | st_cells
    # shellcheck disable=SC2046
    st_run $(portable)
    want_rc 1
    want_out "$why"
    want_v 'result=red'
  done
  [ -z "$WHY" ]
}

case_hold() {
  st_case "hold$SUF"
  local i
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/.* hold=\([a-z]*\) .*/\1/p' 'up up'
  no_hold
  want_out '[hold] '
  # a red stop drops it too
  { echo "$HDR"; echo 'red1|a|a6000||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  no_hold
  # another owner's hold: nothing runs, nothing is written, ours never goes up
  : > "$C/calls"
  : > "$C/holds/bloomery-zzz-hold"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  rm -f "$C/v.sitting"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 75
  want_out "another owner's hold is up"
  no_calls
  no_hold
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written under another owner's hold"
  rm -f "$C/holds/bloomery-zzz-hold"
  # a TERM mid-call: the call is stopped, the verdict says red, the hold comes down
  { echo "$HDR"; echo 'slow1|a|a6000||512|-'; } | st_cells
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_start $(portable)
  for ((i = 0; i < 100; i++)); do
    grep -q 'call profile=slow1' "$C/calls" 2> /dev/null && break
    kill -0 "$ST_PID" 2> /dev/null || break
    sleep 0.1
  done
  grep -q 'hold=up' "$C/calls" || fault "the slow call did not start under the hold"
  kill -TERM "$ST_PID" 2> /dev/null
  wait "$ST_PID"
  RC=$?
  want_rc 143
  no_hold
  want_v 'result=red'
  want_v 'reason=the sitting ended before its last run'
  [ -z "$WHY" ]
}

# The quiet box: after the hold is up and before anything is written or run, the sitting waits until no compile of another
# track runs; a box that stays busy ends it rc 75 with nothing run and no verdict; before each cell the same look, bounded,
# and a cell goes on after the bound.
case_quiet() {
  st_case "quiet$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  # busy for two looks, then quiet: the wait names the compiles, ends, and only then does the sitting write anything
  printf '%s\n' 'rustc:101 cargo:100' 'rustc:101' '-' > "$C/ps.seq"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  want_out '[quiet] waiting: cargo x1, rustc x1'
  want_out '[quiet] box quiet after '
  want_before " up: $C/holds/bloomery-rel-hold" '[quiet] waiting: cargo x1, rustc x1'
  want_before '[quiet] box quiet after ' '[sitting] stage='
  want_before '[sitting] stage=' '=== sitting: round 1, 1 of 1'
  # a box that stays busy: rc 75 naming them, once a minute's line more than once, nothing run, nothing written
  st_fresh
  printf '%s\n' 'rustc:101 cargo:100' > "$C/ps.seq"
  ST_EXTRA=(SITTING_QUIET_WAIT=5)  # whole seconds: a 2 s bound can end after one, too short for six looks under load
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 75
  want_out 'processes of other tracks still run after'
  want_out 'cargo x1, rustc x1 [rustc(101) cargo(100)]'
  [ "$(grep -cF '[quiet] waiting: ' "$C/out")" -ge 2 ] || fault "the wait named the compiles once, want a line every sixth look"
  no_calls
  no_hold
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written by a sitting that never ran"
  # a quiet box before each cell: the compile that starts after cell 1 is waited out before cell 2
  st_fresh
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; echo 'pb|a|a6000||512|-'; } | st_cells
  printf '%s\n' '-' '-' 'rustc:7' '-' > "$C/ps.seq"
  ST_EXTRA=()
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  [ "$(grep -cF '[quiet] waiting: rustc x1' "$C/out")" = 1 ] || fault "the compile before cell 2 was not named exactly once"
  want_before 'round 1, 1 of 2: cell 1 green' '[quiet] waiting: rustc x1'
  want_before '[quiet] waiting: rustc x1' 'round 1, 2 of 2: cell 2: '
  # a compile that outlasts the between-cells bound: the cell goes on, and says so
  st_fresh
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  printf '%s\n' '-' 'rustc:7' > "$C/ps.seq"
  ST_EXTRA=(SITTING_QUIET_WAIT=2)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  want_out '[quiet] between cells: still running after 2 s: rustc x1'
  calls_seq "$CALL_PROF" 'pa pa'
  # a zombie is no compile: the box is quiet at the first look
  st_fresh
  printf '%s\n' 'rustc:9:Z' > "$C/ps.seq"
  ST_EXTRA=(SITTING_QUIET_WAIT=2)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  want_out '[quiet] box quiet after 0 s'
  [ "$(grep -cF '[quiet] waiting: ' "$C/out")" = 0 ] || fault "a zombie was waited for"
  # a ps that does not answer is not a quiet box
  st_fresh
  rm -f "$C/ps.seq"
  : > "$C/ps.fail"
  ST_EXTRA=()
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 2
  want_out 'ps -eo pid=,stat=,comm= does not answer'
  no_calls
  no_hold
  rm -f "$C/ps.fail"
  # a bound that is no number
  st_fresh
  ST_EXTRA=(SITTING_QUIET_WAIT=abc)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 64
  want_out 'SITTING_QUIET_WAIT is whole seconds'
  no_calls
  no_hold
  [ -z "$WHY" ]
}

case_dry() {
  st_case "dry$SUF"
  {
    printf 'file/profile\tplace\tcards\tctx\tD\tchecks\tload_s\tpp\ttg\n'
    printf 'pa=/m/a.gguf\ta\t3090\t\t512,4096\t-\t10\t100\t20\n'
    printf 'pb\tbp\tboth\t\t512\t--client toktape\t20\t50\t10\n'
  } > "$C/cells.tsv"
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_rc 0
  want_out 'round 1 (family-major'
  want_out 'round 2 (the reverse order'
  want_out '[plan]   1. cell 1: pa=/m/a.gguf place=a cards=3090'
  # cell 1 (4 servers a depth and a round): 4 x (5 + 10) + 2 x (512/100 + 127/20 + 4096/100 + 127/20) = 60 + 117.56; cell 2 (2 servers):
  # 2 x (5 + 20) + 2 x (512/50 + 127/10) + 5855/50 + 3 x 128/10 = 50 + 45.88 + 155.5; the sum 428.94
  want_out 'predicted wall = 429 s (~7 min)'
  want_out '[dry] no hold up, no call, no verdict'
  no_calls
  no_hold
  [ ! -e "$C/v.sitting" ] || fault "a dry run wrote a verdict"
  # --fold depths: cell 1 loads twice (2 x (5 + 10) = 30 + 117.56), cell 2 as before: 147.56 + 251.38 = 398.94
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run --fold depths
  want_rc 0
  want_out 'predicted wall = 399 s (~7 min)'
  { echo "$HDR"; echo 'pa|a|3090||512|--client toktape'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_rc 0
  want_out 'predicted wall >= 10 s'
  want_out 'unpriced (a lower bound): cell 1 load (no load_s, SITTING_LOAD_EST unset); cell 1 requests and checks (no pp and tg)'
  [ -z "$WHY" ]
}

case_cellserr() {
  st_case "cellserr$SUF"
  local row want
  # one malformed file a time, each fault named with its line
  while IFS='#' read -r row want; do
    { echo "$HDR"; echo "$row"; } | st_cells
    st_run --binary tree --dry-run
    want_rc 64
    want_out "$want"
  done << 'EOF'
p|a|3090||512#5 columns, the header has 6
p|x|3090||512|-#place is a, bp or -, got 'x'
p|a|5090||512|-#cards is 3090, a6000 or both, got '5090'
p|bp|3090||512|-#place bp needs cards both
p|a|3090||512,x|-#each D is a positive number, got 'x'
p|a|3090||512,512|-#D names a depth twice
p|a|3090|-5|512|-#ctx (empty: unset) is a positive number, got '-5'
/abs|a|3090||512|-#file/profile is
p|a|3090||512|   #checks is pass_checks.py's arguments
EOF
  printf 'p\ta\t3090\t\t512\t-\n' > "$C/cells.tsv"
  st_run --binary tree --dry-run
  want_rc 64
  want_out 'the header is'
  echo "$HDR" | st_cells
  st_run --binary tree --dry-run
  want_rc 64
  want_out 'no cell: the header is the only line'
  : > "$C/cells.tsv"
  st_run --binary tree --dry-run
  want_rc 64
  want_out 'no header line'
  [ -z "$WHY" ]
}

case_env() {
  st_case "env$SUF"
  { echo "$HDR"; echo 'pa=/m/a.gguf|-|both||512|-'; echo 'pb|a|3090|4300|512|-'; echo 'pc|a|a6000||512|-'; } | st_cells
  ST_EXTRA=(DEPTH_SERVE_PLACE=bp DEPTH_SERVE_CTX=999 BLOOMERY_REF_MODEL=/x BLOOMERY_REF_MODEL_PROFILE=zz BLOOMERY_TIMING_GPU=GPU-zzz DEPTH_SERVE_TARBALL=/y)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq '1s/ arms=.*//p' 'call profile=pa ref=/m/a.gguf refprof=unset place=unset ctx=unset cards=a6000+3090 gpu=unset tarball='"$C"'/tb/'"$TBN"'.tar.gz after=unset rounds=1 userplan=unset hold=up gpuhold=up:3090,a6000 membase=GPU-bbbb=100,GPU-aaaa=100 fold=off'
  calls_seq '2s/ arms=.*//p' 'call profile=pb ref=unset refprof=unset place=a ctx=4300 cards=unset gpu=GPU-bbbb tarball='"$C"'/tb/'"$TBN"'.tar.gz after=unset rounds=1 userplan=note hold=up gpuhold=up:3090,a6000 membase=GPU-bbbb=100,GPU-aaaa=100 fold=off'
  calls_seq '3s/ arms=.*//p' 'call profile=pc ref=unset refprof=unset place=a ctx=unset cards=unset gpu=unset tarball='"$C"'/tb/'"$TBN"'.tar.gz after=unset rounds=1 userplan=note hold=up gpuhold=up:3090,a6000 membase=GPU-bbbb=100,GPU-aaaa=100 fold=off'
  [ -z "$WHY" ]
}

case_drift() {
  st_case "drift$SUF"
  { echo "$HDR"; echo 'okz|bp|both||512|-'; echo 'drift1|bp|both||512,4096|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  want_out 'plan drifted: round 1 ran stubtree@portable/place=bp/ctx=16384, round 2 stubtree@portable/place=bp/ctx=8192'
  # round 2 runs the reverse order, so the red drift1 is its first cell and okz is not run
  calls_seq "$CALL_PROF" 'okz drift1 drift1'
  want_v 'round 2 cell 1 not run'
  [ -z "$WHY" ]
}

# The fold: the sitting's --fold is every cell's DEPTH_SERVE_FOLD unless the cell's own fold column says otherwise; the plan prices
# the loads either way, the verdict names the fold, and a bad value is refused by name.
case_fold() {
  st_case "fold$SUF"
  local hdr='file/profile|place|cards|ctx|D|checks|load_s|pp|tg|fold'
  { echo "$hdr"; echo 'fa|bp|both||512,4096|-||||'; echo 'fb|a|3090||512,4096|-||||off'; echo 'fc|a|3090||512,4096|-||||depths'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --fold depths
  want_rc 0
  calls_seq 's/.* fold=\([^ ]*\) .*/\1/p' 'depths off depths depths off depths'
  grep -qx 'fold=depths' "$C/v.sitting" || fault "the verdict's header lacks fold=depths"
  want_v 'cell 2: fb place=a cards=3090 ctx=unset D=512,4096 checks=- fold=off'
  want_v '== fold audit over the sitting'
  want_v 'fold cell 1 D=512 | tg -5.00 % | pp +0.00 %'
  want_v 'fold pooled over 4 pairs | tg -5.00 % ± 0.00 % (sd 0.00 %) [clears 0] | pp +0.00 % ± 0.00 % (sd 0.00 %) [holds 0]'
  grep -q 'fold cell 2 ' "$C/v.sitting" && fault "a cell with fold=off is in the fold audit"
  st_fresh
  # shellcheck disable=SC2046
  st_run $(portable) --fold off
  want_rc 0
  calls_seq 's/.* fold=\([^ ]*\) .*/\1/p' 'off off depths depths off off'
  grep -qx 'fold=off' "$C/v.sitting" || fault "the verdict's header lacks fold=off"
  want_v 'cell 1: fa place=bp cards=both ctx=unset D=512,4096 checks=- fold=off'
  want_v 'cell 3: fc place=a cards=3090 ctx=unset D=512,4096 checks=- fold=depths'
  want_v 'fold pooled over 2 pairs'
  grep -q 'fold cell 1 ' "$C/v.sitting" && fault "a cell under --fold off is in the fold audit"
  # the plan: 8 loads under --fold depths (fa 2, fb 4, fc 2), 12 with every fold off, and 10 under --fold off (fc's own stays)
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run --fold depths
  want_rc 0
  want_out '[plan] loads: 8 servers (a cell with fold=depths loads one a round for all its depths; 12 with every cell'"'"'s fold off)'
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run --fold off
  want_out '[plan] loads: 10 servers'
  # shellcheck disable=SC2046
  st_run $(portable) --fold sometimes
  want_rc 64
  want_out "--fold is off or depths, got 'sometimes'"
  { echo "$hdr"; echo 'fa|bp|both||512|-||||maybe'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 64
  want_out "fold is off or depths (empty: the sitting's --fold), got 'maybe'"
  # a third depth would be the first request of neither round, so a folded cell has two at most: refused by name, whether the
  # cell says fold=depths or takes the sitting's --fold depths; fold=off keeps any number of depths, and so does the default
  { echo "$hdr"; echo 'fa|bp|both||512,2048,4096|-||||depths'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_rc 64
  want_out "fold=depths with 3 depths (512,2048,4096): the sitting runs 2 rounds, so only 2 depths are ever the first request of a folded server"
  { echo "$hdr"; echo 'fa|bp|both||512,2048,4096|-||||'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run --fold depths
  want_rc 64
  want_out "fold=depths with 3 depths"
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run --fold off
  want_rc 0
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_rc 0
  { echo "$hdr"; echo 'fa|bp|both||512,2048,4096|-||||off'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_rc 0
  [ -z "$WHY" ]
}

# The default is no fold: the release's headline rows are fresh servers' first requests. No --fold flag gives every cell without
# a fold column of its own fold=off in the call and in the verdict, and a cell's own fold=depths still wins.
case_folddefault() {
  st_case "folddefault$SUF"
  local hdr='file/profile|place|cards|ctx|D|checks|load_s|pp|tg|fold'
  { echo "$hdr"; echo 'fa|bp|both||512,4096|-||||'; echo 'fb|a|3090||512,4096|-||||'; echo 'fc|a|3090||512,4096|-||||depths'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/.* fold=\([^ ]*\) .*/\1/p' 'off off depths depths off off'
  grep -qx 'fold=off' "$C/v.sitting" || fault "the verdict's header lacks fold=off"
  want_v 'cell 1: fa place=bp cards=both ctx=unset D=512,4096 checks=- fold=off'
  want_v 'cell 3: fc place=a cards=3090 ctx=unset D=512,4096 checks=- fold=depths'
  want_v 'fold pooled over 2 pairs'
  grep -q 'fold cell 1 ' "$C/v.sitting" && fault "a cell on the default fold is in the fold audit"
  # shellcheck disable=SC2046
  st_run $(portable) --dry-run
  want_out '[plan] loads: 10 servers'
  [ -z "$WHY" ]
}

# The fold audit in the shape of the 0.3.13 validation sitting: two depths a cell, one round a depth-serve call, two sitting
# rounds. A call's own tables hold no first row of the depth that is second on its server, and the sitting printed nothing else;
# round 2 rotates the arms, so each depth is first in one round and shared in the other, and the audit pairs them over the rounds.
# The pairs are on the console as well as in the verdict. A depth without a pair is named with the rows it has: fa pairs both
# depths; cold1's first requests are [cold]; fb is red in round 1 (a check FAIL, its rows printed) so its round 2 did not run; red1
# is red before any row; fc has one depth, so nothing of it was shared.
case_foldshape() {
  st_case "foldshape$SUF"
  { echo "$HDR"; echo 'fa|bp|both||512,4096|-'; echo 'cold1|a|3090||512,4096|-'; echo 'fb|a|3090||512,4096|--stub-failline'; echo 'red1|a|3090||512,4096|-'; echo 'fc|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --fold depths --keep-going
  want_rc 1
  calls_seq '1s/.* arms=//p' 'stubtree@portable:512 stubtree@portable:4096'
  calls_seq '$s/.* arms=//p' 'stubtree@portable:4096 stubtree@portable:512'
  local line
  for line in '== fold audit over the sitting' \
    'fold cell 1 D=512 | tg -5.00 % | pp +0.00 %' 'fold cell 1 D=4096 | tg -5.00 % | pp +0.00 %' \
    'fold cell 2 D=512 | no pair: first rows 0 warm + 1 [cold], shared rows 1 warm + 0 [cold]' \
    'fold cell 2 D=4096 | no pair: first rows 0 warm + 1 [cold], shared rows 1 warm + 0 [cold]' \
    'fold cell 3 D=512 | no pair: first rows 1 warm + 0 [cold], shared rows 0 warm + 0 [cold]' \
    'fold cell 3 D=4096 | no pair: first rows 0 warm + 0 [cold], shared rows 1 warm + 0 [cold]' \
    'fold cell 4 D=512 | no pair: no row of this depth (no run of the cell printed one)' \
    'fold cell 4 D=4096 | no pair: no row of this depth (no run of the cell printed one)' \
    'fold pooled over 2 pairs | tg -5.00 % ± 0.00 % (sd 0.00 %) [clears 0] | pp +0.00 % ± 0.00 % (sd 0.00 %) [holds 0]'; do
    want_out "$line"
    want_v "$line"
  done
  grep -q 'fold cell 5 ' "$C/out" "$C/v.sitting" && fault "a folded cell of one depth is in the fold audit"
  want_before '== fold audit over the sitting' 'sitting: result='
  [ -z "$WHY" ]
}

case_busy() {
  st_case "busy$SUF"
  { echo "$HDR"; echo 'busy1|bp|both||512,4096|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  want_out '[cpu-busy]: a co-tenant voided them'
  [ -z "$WHY" ]
}

case_rows() {
  st_case "rows$SUF"
  { echo "$HDR"; echo 'short1|bp|both||512,4096|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  want_out '1 ROW lines, the cell has 2 arms'
  [ -z "$WHY" ]
}

case_dryenv() {
  st_case "dryenv$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  ST_EXTRA=(BLOOMERY_DRY=1)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 64
  want_out 'BLOOMERY_DRY is set'
  no_calls
  [ -z "$WHY" ]
}

case_relpaths() {
  st_case "relpaths$SUF"
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  # the case directory is the working directory: every path below is relative to it
  st_run --binary portable --cells cells.tsv --tarball "tb/$TBN.tar.gz" --out rel.sitting
  want_rc 0
  want_out "sitting: verdict $C/rel.sitting"
  want_out "sitting: work $C/rel.sitting.d"
  [ -s "$C/rel.sitting" ] || fault "no verdict at the path the run printed"
  grep -qF "tarball=$C/tb/$TBN.tar.gz" "$C/rel.sitting" 2> /dev/null || fault "the verdict names the tarball by a relative path"
  grep -qF "cells_file=$C/cells.tsv" "$C/rel.sitting" 2> /dev/null || fault "the verdict names the cells file by a relative path"
  [ -z "$WHY" ]
}

case_archive() {
  st_case "archive$SUF"
  local n
  { echo "$HDR"; echo 'pa|a|a6000||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  # shellcheck disable=SC2046
  st_run $(portable)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  n=$(find "$C" -maxdepth 1 -name 'v.sitting.*' ! -name v.sitting.d | wc -l | tr -d ' ')
  [ "$n" = 4 ] || fault "$n earlier verdicts and directories kept, want 4"
  [ -z "$WHY" ]
}

# The GPU hold: gpu-gate.sh's sitting hold names the cards the cells time for as long as the sitting runs. It goes up after the box
# hold and before the card look, and comes down on every exit; another sitting's live hold ends this one rc 75 with nothing run.
case_gpuhold() {
  st_case "gpuhold$SUF"
  local dp k
  { echo "$HDR"; echo 'pa|a|3090||512|-'; echo 'pb|a|a6000||512|-'; echo 'pc|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/.* gpuhold=\([^ ]*\) .*/\1/p' 'up:3090,a6000 up:3090,a6000 up:3090,a6000 up:3090,a6000 up:3090,a6000 up:3090,a6000'
  want_out 'sitting hold '
  want_before " up: $C/holds/bloomery-rel-hold" 'sitting hold '
  want_before 'sitting hold ' '[quiet] box quiet after '
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after a green sitting"
  no_hold
  # only the A6000 is timed: only the A6000 is held
  st_fresh
  { echo "$HDR"; echo 'pb|a|a6000||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/.* gpuhold=\([^ ]*\) .*/\1/p' 'up:a6000 up:a6000'
  # a red cell ends the sitting, and the hold comes down with it
  st_fresh
  { echo "$HDR"; echo 'red1|a|a6000||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after a red sitting"
  # another sitting's live hold: rc 75, nothing runs or is written, its hold stays
  st_fresh
  { echo "$HDR"; echo 'pa|a|3090||512|-'; } | st_cells
  bash "$C/tree/tools/gpu-gate.sh" --test-locks "$C/stub" --sitting-hold up other 3090 $$ > /dev/null 2>&1
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 75
  want_out 'is up for owner other'
  want_out 'the GPU hold of the cards the cells time (3090) did not go up'
  no_calls
  no_hold
  grep -q 'owner=other ' "$C/stub/sitting.gpuhold" 2> /dev/null || fault "the other sitting's hold is gone"
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written by a sitting that never ran"
  # a dead hold (its sitting is gone) is taken over
  rm -f "$C/stub/sitting.gpuhold"
  sleep 0 &
  dp=$!
  wait "$dp"
  printf 'owner=gone cards=3090 pid=%s start=x since=1\n' "$dp" > "$C/stub/sitting.gpuhold"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  want_out 'is dead: taking it over'
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after the sitting that took a dead one over"
  # a TERM mid-call: the hold comes down
  { echo "$HDR"; echo 'slow1|a|a6000||512|-'; } | st_cells
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_start $(portable)
  for ((k = 0; k < 100; k++)); do
    grep -q 'call profile=slow1' "$C/calls" 2> /dev/null && break
    kill -0 "$ST_PID" 2> /dev/null || break
    sleep 0.1
  done
  grep -q 'gpuhold=up:a6000' "$C/calls" || fault "the slow call did not start under the GPU hold"
  kill -TERM "$ST_PID" 2> /dev/null
  wait "$ST_PID"
  RC=$?
  want_rc 143
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after a TERM"
  [ -z "$WHY" ]
}

# The memory baseline: read once at the sitting's start, with the cards idle (after the card look), for each card the cells time, and
# handed to every call; a card that is still releasing memory is read until it is at rest, else the sitting ends rc 75.
case_baseline() {
  st_case "baseline$SUF"
  { echo "$HDR"; echo 'pa|a|3090||512|-'; echo 'pb|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq 's/.* membase=\([^ ]*\) .*/\1/p' 'GPU-bbbb=100,GPU-aaaa=100 GPU-bbbb=100,GPU-aaaa=100 GPU-bbbb=100,GPU-aaaa=100 GPU-bbbb=100,GPU-aaaa=100'
  want_out '[baseline] '
  want_before '[quiet] box quiet after ' '[baseline] '
  want_before '[baseline] ' '=== sitting: round 1, 1 of 2'
  # a card still releasing memory is no baseline: reads that differ by more than 64 MiB are read again
  st_fresh
  { echo "$HDR"; echo 'pa|a|3090||512|-'; } | st_cells
  printf '5000\n3000\n100\n100\n' > "$C/mem.GPU-bbbb.seq"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  calls_seq '1s/.* membase=\([^ ]*\) .*/\1/p' 'GPU-bbbb=100'
  # a card that never comes to rest: rc 75 naming it, nothing run, nothing held
  st_fresh
  rm -f "$C/mem.GPU-bbbb.i"
  awk 'BEGIN { for (i = 0; i < 400; i++) print (i % 2 ? 2000 : 100) }' > "$C/mem.GPU-bbbb.seq"
  ST_EXTRA=(SITTING_BASELINE_WAIT=1)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 75
  want_out "the card GPU-bbbb's used memory is still changing after"
  no_calls
  no_hold
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after a sitting that never ran"
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written by a sitting that never ran"
  # a card that gives no reading is no baseline either
  st_fresh
  rm -f "$C/mem.GPU-bbbb.i"
  printf 'x\n' > "$C/mem.GPU-bbbb.seq"
  ST_EXTRA=(SITTING_BASELINE_WAIT=2)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 69
  want_out 'gives no used-memory reading'
  no_calls
  no_hold
  [ -z "$WHY" ]
}

# The card look: a program on a card the cells time, running when the sitting begins (a gate that began before the holds), is waited
# for and named, never signalled; one on another card is not.
case_cardlook() {
  st_case "cardlook$SUF"
  local k
  { echo "$HDR"; echo 'pa|a|3090||512|-'; } | st_cells
  printf '%s\n' '4242:gate_foo:22558:GPU-bbbb' '4242:gate_foo:22558:GPU-bbbb' '-' > "$C/gpu.seq"
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  want_out '[quiet] waiting: gpu:gate_foo x1'
  want_out '[quiet] box quiet after '
  want_before 'sitting hold ' '[quiet] waiting: gpu:gate_foo x1'
  want_before '[quiet] waiting: gpu:gate_foo x1' '[baseline] '
  # a program on a card the cells do not time is none of the sitting's business
  st_fresh
  rm -f "$C/gpu.i"
  { echo "$HDR"; echo 'pb|a|a6000||512|-'; } | st_cells
  printf '%s\n' '4242:gate_foo:22558:GPU-bbbb' > "$C/gpu.seq"
  ST_EXTRA=(SITTING_QUIET_WAIT=5)  # a sitting that waits for it anyway ends rc 75 here, not after the default 30 minutes
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  [ "$(grep -cF '[quiet] waiting: ' "$C/out")" = 0 ] || fault "a program on a card the cells do not time was waited for"
  # a card that stays busy: rc 75 naming the program, nothing run, both holds down
  st_fresh
  rm -f "$C/gpu.i"
  { echo "$HDR"; echo 'pa|a|3090||512|-'; } | st_cells
  printf '%s\n' '4242:gate_foo:22558:GPU-bbbb' > "$C/gpu.seq"
  ST_EXTRA=(SITTING_QUIET_WAIT=3)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 75
  want_out 'processes of other tracks still run after'
  want_out 'gpu:gate_foo x1 [gpu:gate_foo(4242)]'
  no_calls
  no_hold
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "the GPU hold is still up after a sitting that never ran"
  [ ! -e "$C/v.sitting" ] || fault "a verdict was written by a sitting that never ran"
  # a compute list that does not answer is not an idle card
  st_fresh
  rm -f "$C/gpu.seq" "$C/gpu.i"
  : > "$C/gpu.fail"
  ST_EXTRA=()
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 2
  want_out 'nvidia-smi --query-compute-apps does not answer'
  no_calls
  no_hold
  # a gate that holds the 3090's lock without a compute context yet is waited for, named by its card; a held lock of a card the cells
  # do not time is not
  rm -f "$C/gpu.fail"
  : > "$C/locks/bloomery-gate.lock"
  PATH="$C/bin:$PATH" flock "$C/locks/bloomery-gate.lock" sh -c 'i=0; while [ ! -e "$1" ] && [ "$i" -lt 600 ]; do sleep 0.05; i=$((i + 1)); done' _ "$C/release-gate" &
  echo "$!" >> "$T/pids"
  sleep 0.5
  st_fresh
  { echo "$HDR"; echo 'pb|a|a6000||512|-'; } | st_cells
  ST_EXTRA=(SITTING_QUIET_WAIT=5)
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 0
  [ "$(grep -cF '[quiet] waiting: ' "$C/out")" = 0 ] || fault "a gate lock of a card the cells do not time was waited for"
  st_fresh
  { echo "$HDR"; echo 'pa|a|3090||512|-'; } | st_cells
  ST_EXTRA=()
  # shellcheck disable=SC2046
  st_start $(portable)
  for ((k = 0; k < 200; k++)); do
    grep -qF '[quiet] waiting: gate-lock:3090 x1' "$C/out" 2> /dev/null && break
    kill -0 "$ST_PID" 2> /dev/null || break
    sleep 0.1
  done
  : > "$C/release-gate"
  wait "$ST_PID"
  RC=$?
  want_rc 0
  want_out '[quiet] waiting: gate-lock:3090 x1'
  want_out '[quiet] box quiet after '
  [ -z "$WHY" ]
}

# blk <verdict> <header prefix>: the run block that starts with that header, as the verdict holds it.
blk() { awk -v h="$2" 'index($0, h) == 1 { f = 1; print; next } f && /^$/ { exit } f' "$1"; }
# blk_pid <verdict> <header prefix>: the pid of the sitting that wrote the block (its stamp's), empty when it has none.
blk_pid() { blk "$1" "$2" | sed -n 's/^stamp .* pid=//p'; }

# A resumed sitting: --resume VERDICT runs only the runs the verdict has no green block for (or, with --only, every run of the named
# cells), on the same binary, and writes one merged verdict whose other blocks keep their stamps.
case_resume() {
  st_case "resume$SUF"
  local p1 p2 first
  { echo "$HDR"; echo 'ok1|bp|both||512|--stub-pass'; echo 'flaky1|bp|both||512|-'; echo 'ok2|bp|both||512|-'; } | st_cells
  # the first sitting: flaky1 is red in round 1, --keep-going runs ok2, and round 2 runs the cells whose round 1 was green
  # shellcheck disable=SC2046
  st_run $(portable) --keep-going
  want_rc 1
  calls_seq "$CALL_PROF" 'ok1 flaky1 ok2 ok2 ok1'
  want_v 'result=red'
  want_v 'stamp finished='
  want_v 'round 2 cell 2 not run'
  cp "$C/v.sitting" "$C/first.sitting"
  # the resume: flaky1's round 1 and its round 2, nothing else
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting"
  want_rc 0
  calls_seq "$CALL_PROF" 'flaky1 flaky1'
  calls_seq 's/.* membase=\([^ ]*\) .*/\1/p' 'GPU-bbbb=100,GPU-aaaa=100 GPU-bbbb=100,GPU-aaaa=100'
  want_out '[resume] '
  want_v 'result=green'
  want_v 'resumed_runs=round 1 cells 2; round 2 cells 2'
  want_v "resumed_from=$C/v.sitting."
  want_v '== round 1 cell 2 rc=0 result=green'
  want_v '== round 2 cell 2 rc=0 result=green'
  want_v '== round 2 cell 1 rc=0 result=green'
  want_v '== cell 2 means over its rows'
  # the untouched runs keep their blocks, stamps included, byte for byte; the re-run ones have a stamp of their own
  first=$(blk "$C/first.sitting" '== round 1 cell 1 rc=0')
  [ -n "$first" ] || fault "the first verdict has no round 1 cell 1 block"
  [ "$first" = "$(blk "$C/v.sitting" '== round 1 cell 1 rc=0')" ] || fault "round 1 cell 1's block changed in the merge"
  [ "$(blk "$C/first.sitting" '== round 1 cell 3 rc=0')" = "$(blk "$C/v.sitting" '== round 1 cell 3 rc=0')" ] || fault "round 1 cell 3's block changed in the merge"
  [ "$(blk "$C/first.sitting" '== round 2 cell 1 rc=0')" = "$(blk "$C/v.sitting" '== round 2 cell 1 rc=0')" ] || fault "round 2 cell 1's block changed in the merge"
  p1=$(blk_pid "$C/v.sitting" '== round 1 cell 1 rc=0')
  p2=$(blk_pid "$C/v.sitting" '== round 1 cell 2 rc=0')
  { [ -n "$p1" ] && [ -n "$p2" ] && [ "$p1" != "$p2" ]; } || fault "the stamps of the kept run ('$p1') and the re-run one ('$p2') are not two sittings'"
  [ "$(blk_pid "$C/v.sitting" '== round 2 cell 2 rc=0')" = "$p2" ] || fault "round 2 cell 2 was not stamped by the resuming sitting"
  [ "$(blk_pid "$C/v.sitting" '== round 2 cell 1 rc=0')" = "$p1" ] || fault "round 2 cell 1 lost the first sitting's stamp"
  [ -e "$C/v.sitting.d" ] || fault "no work directory for the resumed sitting"
  # the merged verdict is a sitting's verdict: green, nothing left to resume
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting"
  want_rc 64
  want_out 'nothing to resume'
  no_calls
  no_hold
  # --only re-runs the named cells in both rounds, green or not, and keeps the rest as they were
  cp "$C/v.sitting" "$C/second.sitting"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting" --only 1
  want_rc 0
  calls_seq "$CALL_PROF" 'ok1 ok1'
  want_v 'resumed_runs=round 1 cells 1; round 2 cells 1'
  [ "$(blk "$C/second.sitting" '== round 1 cell 3 rc=0')" = "$(blk "$C/v.sitting" '== round 1 cell 3 rc=0')" ] || fault "an unnamed cell's block changed under --only"
  [ "$(blk_pid "$C/v.sitting" '== round 1 cell 1 rc=0')" != "$p1" ] || fault "the named cell kept its old stamp"
  [ -z "$WHY" ]
}

# What a resume refuses, before any hold goes up: another binary (65), another stage, another cells file, a cell the file does not
# have, --only alone; a kept red keeps the merged verdict red; a kept round 1 still checks round 2's plan.
case_resumeguard() {
  st_case "resumeguard$SUF"
  { echo "$HDR"; echo 'ok1|bp|both||512|-'; echo 'red1|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --keep-going
  want_rc 1
  cp "$C/v.sitting" "$C/first.sitting"
  : > "$C/calls"
  # a block run on another binary than the header says
  sed '/^stamp /s/binary_sha256=[0-9a-f]*/binary_sha256=deadbeef/' "$C/first.sitting" > "$C/tampered.sitting"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/tampered.sitting"
  want_rc 65
  want_out 'was run on binary sha256 deadbeef'
  no_calls
  no_hold
  # another stage
  st_run --binary tree --resume "$C/first.sitting"
  want_rc 64
  want_out 'is a stage=portable verdict'
  no_calls
  # another cells file (one the guard alone keeps from running a cell), and another fold: the fold is part of a cell
  { echo "$HDR"; echo 'ok1|bp|both||512|-'; echo 'ok3|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/first.sitting"
  want_rc 64
  want_out "are not this cells file's"
  no_calls
  { echo "$HDR"; echo 'ok1|bp|both||512|-'; echo 'red1|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/first.sitting" --fold depths
  want_rc 64
  want_out "and the --fold it was run with"
  no_calls
  { echo "$HDR"; echo 'ok1|bp|both||512|-'; echo 'red1|bp|both||512|-'; } | st_cells
  # a cell the file does not have, a number that is none, --only without --resume
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/first.sitting" --only 9
  want_rc 64
  want_out '--only names cell 9, and the cells file has 2 cells'
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/first.sitting" --only x
  want_rc 64
  want_out "--only takes cell numbers, 2,5 or 3-7, got 'x'"
  # shellcheck disable=SC2046
  st_run $(portable) --only 1
  want_rc 64
  want_out '--only names the cells --resume re-runs'
  no_calls
  # a verdict that is none
  echo 'nonsense' > "$C/none.sitting"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/none.sitting"
  want_rc 64
  want_out "it is no verdict of this script"
  # a kept red keeps the merged verdict red: --only 1 re-runs ok1 and leaves red1's round 1 as it was
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/first.sitting" --only 1
  want_rc 1
  calls_seq "$CALL_PROF" 'ok1 ok1'
  want_v 'result=red'
  want_v '== round 1 cell 2 rc=1 result=red'
  want_v '== round 2 cell 2 not run'
  want_out 'sitting: red: round 1 cell 2 (red1 place=bp cards=both)'
  # another binary: rc 65, nothing runs, the verdict is not touched. A verdict without stamps (an earlier script's) has the header's
  # sha256 alone to go by.
  printf '#!/bin/sh\necho "bloomery-serve 9.9.9 (commit other)"\n' > "$C/home/repo/stubtree/target/portable/release/bloomery-serve"
  cp "$C/home/repo/stubtree/target/portable/release/bloomery-serve" "$C/tb/$TBN/bin/bloomery-serve"
  tar -czf "$C/tb/$TBN.tar.gz" -C "$C/tb" "$TBN"
  grep -v '^stamp ' "$C/first.sitting" > "$C/stampless.sitting"
  cp "$C/stampless.sitting" "$C/v.sitting"
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting"
  want_rc 65
  want_out 'the binary here (sha256 '
  want_out 'never mixes binaries'
  no_calls
  no_hold
  cmp -s "$C/v.sitting" "$C/stampless.sitting" || fault "the refused resume touched the verdict"
  [ -z "$WHY" ]
}

# A kept round 1 still checks round 2's plan against its labels.
case_resumedrift() {
  st_case "resumedrift$SUF"
  { echo "$HDR"; echo 'drift1|bp|both||512|-'; echo 'flaky1|bp|both||512|-'; } | st_cells
  # the first sitting stops at flaky1's red: drift1's round 1 is green and its round 2 has not run
  # shellcheck disable=SC2046
  st_run $(portable)
  want_rc 1
  calls_seq "$CALL_PROF" 'drift1 flaky1'
  # the resume runs flaky1's round 1, then round 2: drift1's second call prints another context than its kept round 1
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting"
  want_rc 1
  want_out 'plan drifted: round 1 ran stubtree@portable/place=bp/ctx=16384, round 2 stubtree@portable/place=bp/ctx=8192'
  [ -z "$WHY" ]
}

# A dry run of a resume names its runs and touches nothing.
case_resumedry() {
  st_case "resumedry$SUF"
  { echo "$HDR"; echo 'ok1|bp|both||512|-'; echo 'red1|bp|both||512|-'; } | st_cells
  # shellcheck disable=SC2046
  st_run $(portable) --keep-going
  cp "$C/v.sitting" "$C/first.sitting"
  : > "$C/calls"
  # shellcheck disable=SC2046
  st_run $(portable) --resume "$C/v.sitting" --dry-run
  want_rc 0
  want_out '[dry] resume '
  want_out 'runs round 1 cells 2; round 2 cells 2'
  no_calls
  no_hold
  cmp -s "$C/v.sitting" "$C/first.sitting" || fault "a dry run touched the verdict"
  [ ! -e "$C/stub/sitting.gpuhold" ] || fault "a dry run put the GPU hold up"
  [ -z "$WHY" ]
}

# The jobs: every case on the script, and for each guard the case that guards it on a copy of the script with that one
# line changed, which must fail. A job runs in a subshell of its own (its directory and globals are its own) and writes its
# report to $T/res.<k>; the driver runs them in parallel batches and prints the reports in order.
JOBS_KIND=() JOBS_ARG=() MUT_NAME=() MUT_CASE=() MUT_OLD=() MUT_NEW=()

# mutant <name> <case> <old> <new>
mutant() {
  MUT_NAME+=("$1") MUT_CASE+=("$2") MUT_OLD+=("$3") MUT_NEW+=("$4")
  JOBS_KIND+=(mutant) JOBS_ARG+=("$((${#MUT_NAME[@]} - 1))")
}

# job_report <name> <why>: a failed job's line and the tail of its sitting's output.
job_report() {
  echo "FAIL $1: $2"
  [ ! -r "$C/out" ] || sed 's/^/    | /' "$C/out" | head -n 30
}

# job_run <k>: the k-th job, its report on stdout.
job_run() {
  local k=$1 arg=${JOBS_ARG[$1]} f
  if [ "${JOBS_KIND[k]}" = case ]; then
    SUT=$self SUF=
    if "case_$arg"; then echo "ok $arg"; else job_report "$arg" "${WHY#; }"; fi
    return 0
  fi
  f=$T/mutant-${MUT_NAME[arg]}.sh
  if ! python3 -I - "$self" "$f" "${MUT_OLD[arg]}" "${MUT_NEW[arg]}" << 'PY'; then
import sys

src, dst, old, new = sys.argv[1:5]
s = open(src).read()
# The mutant calls below hold the old text too: only the code above the self-test is searched.
cut = s.index("\n# ---- --self-test ")
head, tail = s[:cut], s[cut:]
if head.count(old) != 1:
    sys.exit(1)
open(dst, "w").write(head.replace(old, new) + tail)
PY
    echo "FAIL mutant-${MUT_NAME[arg]}: its line is not in the script exactly once: ${MUT_OLD[arg]}"
    return 0
  fi
  SUT=$f SUF=-m-${MUT_NAME[arg]}
  if "case_${MUT_CASE[arg]}"; then
    job_report "mutant-${MUT_NAME[arg]}" "case ${MUT_CASE[arg]} stayed green with the line changed to: ${MUT_NEW[arg]}"
  else
    echo "ok mutant-${MUT_NAME[arg]} turns ${MUT_CASE[arg]} red"
  fi
}

selftest() {
  local self=${BASH_SOURCE[0]} cs n=0 fails=0 pid k first last
  local -a pids
  command -v python3 > /dev/null || { echo "sitting.sh --self-test: python3 is not on PATH" >&2; exit 2; }
  # shellcheck disable=SC2046 # the names are words
  unset BLOOMERY_DRY BLOOMERY_HOLD_OWNER BLOOMERY_LEASE_CARD BLOOMERY_BOX_ENV BLOOMERY_ARM_BOUND BLOOMERY_TIMING_GPU \
    BLOOMERY_TIMING_CARDS BLOOMERY_REF_MODEL BLOOMERY_REF_MODEL_PROFILE BLOOMERY_MODEL $(compgen -v SITTING_) $(compgen -v DEPTH_SERVE_)
  # T0 is the directory mktemp made, the only one this function removes; T is the same directory as `cd` spells it, which
  # is how the paths a case compares are printed by the script.
  T0=$(mktemp -d "${TMPDIR:-/tmp}/sitting-self-test.XXXXXX") || exit 2
  T=$(cd "$T0" && pwd) || exit 2
  : > "$T/pids"
  st_cleanup() {
    while read -r pid; do
      if kill -0 "$pid" 2> /dev/null && ps -o command= -p "$pid" 2> /dev/null | grep -q 'sitting.sh'; then kill "$pid" 2> /dev/null; fi
    done < "$T/pids"
    rm -rf "${T0:?}"
  }
  trap st_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  for cs in order redstop keepgoing verdict tree noportable mismatch checks hold quiet dry cellserr env drift fold busy rows dryenv relpaths archive folddefault foldshape \
    gpuhold baseline cardlook resume resumeguard resumedrift resumedry; do
    JOBS_KIND+=(case) JOBS_ARG+=("$cs")
  done
  mutant family order 'key=lambda c: (first[c["prof"]], PLACE_RANK[c["place"]], c["idx"])' 'key=lambda c: c["idx"]'
  mutant reverse order 'for ((i = total - 1; i >= 0; i--)); do rev+=("${CELLREC[i]}"); done' 'for ((i = 0; i < total; i++)); do rev+=("${CELLREC[i]}"); done'
  mutant rotate order 'k=$(((i + r - 1) % n))' 'k=$i'
  mutant hook-round-2 order 'if [ "$r" = 1 ] && [ "$CF_CHECKS" != - ]; then' 'if [ "$CF_CHECKS" != - ]; then'
  mutant round-2-after-red redstop 'if [ "$ROUND_RED" = 1 ] && [ "$KEEP" = 0 ]; then' 'if false; then'
  mutant round-2-skipped-by-keep keepgoing 'if [ "$ROUND_RED" = 1 ] && [ "$KEEP" = 0 ]; then' 'if [ "$ROUND_RED" = 1 ]; then'
  mutant round-2-red-cell keepgoing '    if round1_green "$CF_IDX"; then
      R2SEL+=("$rec")' '    if true; then
      R2SEL+=("$rec")'
  mutant fold-passed fold ' "DEPTH_SERVE_FOLD=$CF_FOLD"' ''
  mutant fold-cell-wins fold 'c["fold"]]))' '"depths"]))'
  mutant fold-default fold 'fold=fold or default_fold))' 'fold=fold or "off"))'
  mutant fold-audit-called fold '    audit=$(fold_audit 2>&1) || audit="(the fold audit exited $?: $audit)"' '    audit='
  mutant fold-audit-math fold "pct = [100 * (statistics.mean(x[k] for x in d['shared']) / statistics.mean(x[k] for x in d['first']) - 1) for k in (0, 1)]" "pct = [100 * (statistics.mean(x[k] for x in d['first']) / statistics.mean(x[k] for x in d['shared']) - 1) for k in (0, 1)]"
  mutant fold-audit-console foldshape '  if [ -n "$FOLD_AUDIT" ]; then' '  if false; then'
  mutant fold-audit-no-pair foldshape "        if not d['first'] or not d['shared']:" "        if False:"
  mutant fold-audit-folded-cells fold '    [ "$CF_FOLD" != depths ] || out+=' '    out+='
  mutant fold-audit-one-depth foldshape "    if len(deps.split(',')) > 1:" '    if True:'
  mutant fold-depths-cap fold 'and len(dl) > SITTING_ROUNDS:' 'and False:'
  mutant fold-default-off folddefault 'KEEP=0 DRY=0 FOLD=off' 'KEEP=0 DRY=0 FOLD=depths'
  mutant fold-header fold '    echo "fold=$FOLD"' ':'
  mutant fold-flag fold '          off | depths) FOLD=$2 ;;' '          off | depths | *) FOLD=$2 ;;'
  mutant fold-column fold '    if fold not in ("", "off", "depths"):' '    if False:'
  mutant fold-plan fold 'loads = sum(servers_of(c, c["fold"]) for c in cells)' 'loads = sum(servers_of(c, "off") for c in cells)'
  mutant stop-on-red redstop 'if [ "$KEEP" = 0 ]; then STOPPED=' 'if false; then STOPPED='
  mutant tarball-required noportable 'if [ "$STAGE" = portable ] && [ -z "$TARBALL" ]; then' 'if false; then'
  mutant member-sha mismatch '[ "$BIN_SHA" = "$TB_MEMBER_SHA" ] || die 65' 'true || die 65'
  mutant hold-down hold '[ -z "$HOLD_FILE" ] || rm -f "$HOLD_FILE"' ':'
  mutant hold-up hold '  hold_up
  gpu_hold_up' '  :
  gpu_hold_up'
  mutant other-hold hold 'others=$(lease_holds_up "$OWNER") || exit $?' 'others='
  mutant finish-trap hold 'trap finish EXIT' ':'
  mutant quiet-initial quiet '  quiet_wait initial
  baseline_read' '  :
  baseline_read'
  mutant quiet-between quiet '  quiet_wait between' '  :'
  mutant quiet-timeout quiet 'die 75 "processes of other tracks still run' 'echo "processes of other tracks still run'
  mutant quiet-between-goes-on quiet '      if [ "$kind" = initial ]; then
        die 75' '      if true; then
        die 75'
  mutant quiet-zombie quiet '    $2 ~ /^Z/ { next }
' ''
  mutant quiet-ps-fails quiet ' || die 2 "ps -eo pid=,stat=,comm= does not answer: whether the box is quiet cannot be read"' ''
  mutant quiet-wait-number quiet '[[ ${SITTING_QUIET_WAIT:-1800} =~ ^[0-9]+$ ]] || die 64' '[[ ${SITTING_QUIET_WAIT:-1800} =~ ^ ]] || die 64'
  mutant check-fail checks 'elif [ -n "$fails" ]; then' 'elif false; then'
  mutant stage-name tree 'echo "stage=$STAGE"' 'echo "stage=portable"'
  mutant plan-drift drift 'if ! cmp -s "$other" "$labels"; then' 'if false; then'
  mutant busy-rows busy 'if [ "${busy:-0}" != 0 ]; then' 'if false; then'
  mutant row-count rows 'if [ "${rows:-0}" != "$arms" ]; then' 'if false; then'
  mutant env-unset env '-u DEPTH_SERVE_CTX -u DEPTH_SERVE_PLACE -u DEPTH_SERVE_TARBALL' '-u DEPTH_SERVE_TARBALL'
  mutant user-plan env 'envv+=(DEPTH_SERVE_USER_PLAN=note); fi' ':; fi'
  mutant dry-run dry 'if [ "$DRY" = 1 ]; then' 'if false; then'
  mutant absolute-out relpaths '  case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac' '  :'
  mutant absolute-cells relpaths '  case $CELLS in /*) ;; *) CELLS=$PWD/$CELLS ;; esac' '  :'
  mutant absolute-tarball relpaths "  case \$TARBALL in '' | /*) ;; *) TARBALL=\$PWD/\$TARBALL ;; esac" '  :'
  mutant archive archive '  mv "$1" "$to" || die 2 "cannot move the earlier $1 aside"' '  :'
  mutant red-named redstop '"depth-serve.sh exited $rc${fail:+: $fail}"' '"depth-serve.sh exited $rc"'
  mutant gpu-hold-up gpuhold '  gpu_hold_up
  quiet_wait initial' '  :
  quiet_wait initial'
  mutant gpu-hold-down gpuhold '  "${GPU_GATE[@]}" --sitting-hold down "$OWNER" || true' '  :'
  mutant gpu-hold-before-look gpuhold '  gpu_hold_up
  quiet_wait initial' '  quiet_wait initial
  gpu_hold_up'
  mutant gpu-hold-cards gpuhold '  [ "$w3090" = 0 ] || SIT_CARDS=3090' '  SIT_CARDS=3090'
  mutant gpu-hold-refused gpuhold '  [ "$rc" = 0 ] || die "$rc" "the GPU hold of the cards' '  [ "$rc" = 0 ] || true "$rc" "the GPU hold of the cards'
  mutant baseline-passed baseline ' "DEPTH_SERVE_MEM_BASELINE=$MEMBASE"' ''
  mutant baseline-after-look baseline '  quiet_wait initial
  baseline_read' '  baseline_read
  quiet_wait initial'
  mutant baseline-settle baseline '        if [ "$d" -le 64 ]; then break; fi' '        if true; then break; fi'
  mutant baseline-bound baseline "die 75 \"the card \$uuid's used memory is still changing after" "{ break; } # the card \$uuid's used memory is still changing after"
  mutant baseline-no-reading baseline '[ -n "$cur" ] || die 69 "the card $uuid gives no used-memory reading' '[ -n "$cur" ] || true "the card $uuid gives no used-memory reading'
  mutant card-look cardlook '      gprocs=$(quiet_gpu) || die 2 "nvidia-smi --query-compute-apps does not answer: whether the cards the cells time are idle cannot be read"' '      gprocs='
  mutant card-look-fails cardlook ' || die 2 "nvidia-smi --query-compute-apps does not answer: whether the cards the cells time are idle cannot be read"' ''
  mutant card-look-cards cardlook '    NF >= 4 && index(want, " " $4 " ") {' '    NF >= 4 {'
  mutant gate-lock-look cardlook '      glocks=$(quiet_gates)' '      glocks='
  mutant gate-lock-cards cardlook '  for c in ${SIT_CARDS//,/ }; do
    case $c in
      3090) f=' '  for c in 3090 a6000; do
    case $c in
      3090) f='
  mutant stamp resume '    echo "stamp finished=$(now) binary_sha256=$BIN_SHA sitting=$STARTED pid=$$"' '    :'
  mutant resume-select resume '; then return 1; fi
  return 0' '; then return 0; fi
  return 0'
  mutant resume-only resume '    case $ONLYSET in *",$2,"*) return 0 ;; esac' '    :'
  mutant resume-binary resumeguard '    die 65 "the binary here (sha256 $BIN_SHA) is not the one' '    echo "the binary here (sha256 $BIN_SHA) is not the one'
  mutant resume-block-binary resumeguard '      die 65 "round $r cell $i of $RESUME was run' '      echo "round $r cell $i of $RESUME was run'
  mutant resume-stage resumeguard '[ "$RV_STAGE" = "$STAGE" ] || die 64' '[ "$RV_STAGE" = "$STAGE" ] || echo'
  mutant resume-cells resumeguard '[ "$have" = "$exp" ] || die 64' '[ "$have" = "$exp" ] || echo'
  mutant resume-keep resume '      cp "$b" "$WORK/block-r$r-c$CF_IDX.txt" || die 2 "cannot copy the kept run $b into $WORK"' '      :'
  mutant resume-labels resumedrift '| sort -u > "$WORK/labels-r1-c$CF_IDX.txt" || true' '| sort -u > /dev/null || true'
  mutant resume-red-kept resumeguard '        case $head in' '        case "" in'
  mutant resume-notrun resumeguard '        NOTRUN+=("round $r cell $CF_IDX not run: ${STOPPED:-no run of it exists in the verdict being resumed, and this sitting did not run it}")' '        :'
  mutant resume-nothing resume 'die 64 "nothing to resume: every run' 'echo "nothing to resume: every run'
  mutant resume-dry resumedry '        resume_load
        echo "[dry] resume' '        :
        echo "[dry] resume'
  # six jobs at a time: each is a few sittings on stub servers
  for ((first = 0; first < ${#JOBS_KIND[@]}; first += 6)); do
    pids=()
    last=$((first + 5))
    [ "$last" -lt "${#JOBS_KIND[@]}" ] || last=$((${#JOBS_KIND[@]} - 1))
    for ((k = first; k <= last; k++)); do
      job_run "$k" > "$T/res.$k" 2>&1 &
      pids+=("$!")
      echo "$!" >> "$T/pids"
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
  for ((k = 0; k < ${#JOBS_KIND[@]}; k++)); do
    cat "$T/res.$k"
    n=$((n + 1))
    grep -q '^ok ' "$T/res.$k" || fails=$((fails + 1))
  done
  if [ "$fails" = 0 ]; then
    echo "sitting.sh: self-test ok ($n cases)"
    exit 0
  fi
  echo "sitting.sh: self-test: $fails failure(s) in $n cases" >&2
  exit 1
}

case ${1:-} in
  --self-test)
    [ $# = 1 ] || { usage; exit 64; }
    selftest
    ;;
  *)
    main "$@"
    exit $?
    ;;
esac
