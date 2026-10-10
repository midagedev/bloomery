#!/usr/bin/env bash
# shellcheck shell=bash
# The machine-wide lease and what the runners under it share. Sourced, never executed: it defines
# functions and variables and exports nothing, and nothing here may exit or fail at top level
# (most runners run under `set -e`). The one execution is its own `--self-test`, the last block,
# which a sourcing runner never enters.
#
#   source "${BASH_SOURCE[0]%/*}/lease.sh"
#   WITNESS=(head loadavg pressure-io lock-holder model)   # this runner's witness fields, in order
#   lease_take                                              # check the card, wait up to 30 min, or exit
#   witness pre; <the measured run>; witness post
#
# What it owns, so that the runners cannot drift apart:
#   the lease's take: its descriptor (9), the 30-minute wait, exit 75 when the wait runs out;
#   the take's timing-card check (lease_gpu_idle): the machine lock excludes lease runners only, so
#   a compute process that started before the lease is waited out — the lease held — or the run ends
#   at 75, and an nvidia-smi that cannot read the card ends it at 69;
#   the read side, from tools/ref/lease-probe.sh, which this file sources first: the lease's file
#   (LEASE_LOCK), its probe (lease_free, a shared lock), the holds (/root/bloomery-<owner>-hold),
#   tools/box.sh's guard (lease_guard) and naming the lease's holder to a waiter (lease_holders);
#   which process on a card is the runner's own (lease_pid_is_ours): the one rule the card guards
#   share, so a load group's engine between its arms is not read as a co-tenant;
#   the bound on every process a runner starts under the lease (lease_bounded);
#   the card every lease needs (tools/ref/card.py, at lease_take below): no card, no lease;
#   the lease's trace (lease_trace_take, lease_trace_let_go): who took the lease and when it was free;
#   the witness block: its four header forms and every field a runner can list, each spelled once;
#   the CPU-contention guard between arms (guard_cpu), the CPU side of timing-card.sh's guard_other;
#   the two arm helpers the depth and A/B runners share: the LCG prompt of a depth and the
#   bloomery-decode binary of a tree.
# Which card is timed, and the card's own witness lines, stay in timing-card.sh (the `card` field).
#
# Witness fields. `witness <tag>` prints the runner's WITNESS list in order; the first entry is a
# header form, and `indent` makes every field after it start with four spaces. A field that reads a
# runner variable names it in brackets; the runner sets that variable before its first witness.
#   head            --- witness <tag> <utc> ---
#   head-open       --- witness <tag> <utc>
#   head-epoch      --- witness <tag> <utc> epoch <seconds> ---
#   head-load       --- witness <tag> <utc> load=<1 5 15> io=<some avg10> gpu=<utilization per card>
#   card            the timing card's lines: timing-card.sh's witness_card; an apps line of a card
#                   this run times ends with ` [timing-card-busy]` when it lists a process
#   loadavg pressure-cpu pressure-io pressure-io-avg10
#   gpus            index, name, utilization and power of every card
#   gpu-apps        the compute processes on every card
#   stage0-gpus stage0-apps   measure.sh's two tables: every card with memory, the 3090's processes
#   lock-holder     this runner's pid, which holds the lease
#   model           the profile [MODEL_NAME]
#   busiest         the four busiest processes by CPU — a process that ignores the lease shows here only
#   threads         BLOOMERY_THREADS and BLOOMERY_SPIN as the run reads them; empty means the default
#   core core-mhz   the core the run is pinned to and its clock [CORE]
#   cpu cpu-mhz-range   the CPU model and the lowest and highest core clock
#   cpu-freq        every cpu's cpufreq scaling_cur_freq as MHz mean, min and max, and cpu0's governor:
#                   one awk over /sys, no process per core; a reading at the block's instant, not over
#                   the run (timing-card.sh's `card` field prints it too, so every GPU runner has it)
#   meminfo         page cache and free memory, kB
#   mem pgmajfault  available memory and page cache, and the major-fault count
#   read-sectors    sectors read from the model file's block device [MODEL]; the count is machine-wide,
#                   so under the lease the difference between two blocks is what the run paged in
#   table blockstat the engram table's mount and block device, and that device's completed read I/Os
#                   and sectors — the difference between two blocks is what the drive did [DIR SRC DEV STAT]
#   binary          the measured binary and its short sha256 [BIN BIN_SHA]
# The tree this file is in, resolved when it is sourced (a runner may change directory later): the
# root BLOOMERY_LEASE_CARD is relative to, and where card.py lives.
__lease_dir=${BASH_SOURCE[0]%/*}
[ "$__lease_dir" != "${BASH_SOURCE[0]}" ] || __lease_dir=.
LEASE_TREE=$(cd "$__lease_dir/../.." 2> /dev/null && pwd) || LEASE_TREE=
# The lease's file, its probe, the holds, box.sh's guard, lease_holders and now: lease-probe.sh.
# shellcheck source=tools/ref/lease-probe.sh
source "$__lease_dir/lease-probe.sh"
unset __lease_dir

# The lease is descriptor 9 on LEASE_LOCK, held until the runner exits or calls lease_release — and
# by every process that inherited the descriptor, for as long as it lives: a child that outlives its
# runner is still heavy work beside the next sitting, so it keeps the lease and the next runner waits
# rather than records rows beside it. That wait is never anonymous: when the lease is busy the waiter
# names its holder at once (lease_holders), again once a minute, and again when the 30-minute wait
# runs out. A wait that runs out is contention, not a failed measurement: exit 75.
#
# No card, no lease. BLOOMERY_LEASE_CARD names the run's card, docs/cards/<slug>.card relative to
# this tree (tools/ref/card.py has the format and the exit codes). It reaches the box only through
# BLOOMERY_BOX_ENV: tools/box.sh does not carry it otherwise. card.py checks it before the lease file
# is opened and prints its path, sha256 and body as [lease] lines, so the log carries the card ahead
# of the run; a missing or refused card exits with card.py's code (card.py lists them; never 75).
# A runner that runs its arms in rounds sets ROUNDS before lease_take, and an ab or noninf card is checked at
# that count; with no ROUNDS, at BLOOMERY_AB_ROUNDS (the count of a caller that starts this runner
# once per arm, like tools/gpu-ab.py); else at the card's own `rounds`. ROUND_MINUTES, a runner's box
# minutes per round, prices a ruler or margin refusal.
# BLOOMERY_LEASE_LOCK names another lock file, for the Mac stub tests (tools/ref/card-tests/run.sh).
# A run under any other file holds no machine lease, so where /root/bloomery-cpu.lock exists (the box)
# lease_take refuses the override (exit 64): a fake lock during a real sitting would contaminate it.
# A command that tools/ref/lease-hold.sh runs carries BLOOMERY_LEASE_HELD (that script's pid): a lease
# taken inside it would wait on the one already held and end as contention, so it exits 64 at once.
# The record beside the lock names the run's timing cards: TIMING_GPU, and in timing-card.sh's two-card
# mode TIMING_GPU2 after a comma. tools/gpu-gate.sh reads a two-card record as naming neither card, a
# doubt, and so refuses a gate forced onto either card and passes both by under `any`.
# LEASE_CARDS_RECORD says so to timing-card.sh, whose two-card precheck refuses a lease.sh without it.
LEASE_CARDS_RECORD=1
# A compute process on a card this run times voids the numbers: a gate that started before the
# lease (gpu-gate.sh refuses a new gate on the timing card only while the lease is held, so a
# pre-lease process is the hole), a hung one inside the gate runners' 900 s + 10 s bound.
# lease_gpu_idle waits such a process out, polling every LEASE_GPU_POLL s and reprinting the holder
# list every LEASE_GPU_REPORT s, for up to LEASE_GPU_WAIT s (inside which a hung gate ends), the
# lease held while it waits; still busy at the end is contention (75), and an nvidia-smi that is
# missing or fails is a card whose emptiness cannot be read, so the run ends (69). Not BLOOMERY_*
# names — those are the binaries', refused by name when the levers registry does not know them;
# these are the lease's own, and its self-test scales its waits with them.
LEASE_GPU_WAIT=${LEASE_GPU_WAIT:-900}
LEASE_GPU_POLL=${LEASE_GPU_POLL:-15}
LEASE_GPU_REPORT=${LEASE_GPU_REPORT:-60}

# lease_pid_is_ours <pid>: 0 when <pid> is this runner's own process — the one rule the card guards
# share (lease_gpu_idle here, guard_other and guard_cards in timing-card.sh). A load group
# (tools/ref/load-groups.sh) keeps one engine process across a round's arms, resident on the card
# between them, so a guard must tell it from a co-tenant. Two proofs, either closes it: the process
# holds the lease's descriptor 9 — the same file, compared as device and inode, not by path, because
# a path can be recreated where an inode cannot — or it descends from the runner (LEASE_OWNER_PID,
# recorded at the take: $$ in a subshell stays the runner's, so a helper called from one compares
# against the runner). While this runner holds the lease's flock no other runner's process can hold
# that descriptor: an inherited descriptor keeps the lock, so a leftover of an earlier sitting blocks
# the take instead of reading as ours here. A pid whose /proc entry is gone or unreadable cannot be
# proven ours, so it is waited on as foreign; a pid that is not a number is named and waited on the
# same way. BLOOMERY_LEASE_PROC names another /proc tree, as it does for guard_cpu (the self-test,
# which runs where /proc does not exist).
lease_pid_is_ours() {
  local proc=${BLOOMERY_LEASE_PROC:-/proc} owner=${LEASE_OWNER_PID:-$$} ppid id mine seen=0
  case $1 in
    '' | *[!0-9]* | 0)
      echo "[lease] lease_pid_is_ours: a pid is a positive number, got '$1': it cannot be proven ours, so it counts as busy" >&2
      return 1
      ;;
  esac
  if mine=$(__lease_file_id "$proc/$owner/fd/9"); then
    if id=$(__lease_file_id "$proc/$1/fd/9"); then
      [ "$id" = "$mine" ] && return 0
    fi
  fi
  # The ppid walk, bounded as guard_cpu's is: a chain longer than this is not a process tree.
  ppid=$1
  while [ "$ppid" -gt 1 ] && [ "$seen" -lt 256 ]; do
    [ "$ppid" != "$owner" ] || return 0
    ppid=$(__lease_ppid "$proc" "$ppid") || return 1
    seen=$((seen + 1))
  done
  return 1
}
# __lease_file_id <path>: `<device>:<inode>` of the file the path names, following symlinks (a
# /proc fd entry). GNU stat and BSD stat spell it differently, and the self-test runs on both.
__lease_file_id() {
  local id
  id=$(stat -Lc '%d:%i' "$1" 2> /dev/null) || id=$(stat -L -f '%d:%i' "$1" 2> /dev/null) || return 1
  printf '%s' "$id"
}
# __lease_ppid <proc root> <pid>: the pid's ppid from its stat line, read past the comm (which can
# hold spaces and parens: everything through the last ')' is the pid and the comm). 1 when the line
# is gone or is not a stat line.
__lease_ppid() {
  local s
  [ -r "$1/$2/stat" ] || return 1
  read -r s < "$1/$2/stat" || return 1
  s=${s##*)}
  set -- $s
  case ${2:-} in
    '' | *[!0-9]*) return 1 ;;
  esac
  printf '%s' "$2"
}
# __lease_apps_split_own <rows> <pid field>: nvidia-smi compute-apps csv rows split by
# lease_pid_is_ours — the runner's own rows into LEASE_APPS_OWN, the rest (foreign, or a pid that
# cannot be proven ours) into LEASE_APPS_FOREIGN, each row unchanged. The pid field is 2 in
# guard_cards's gpu_uuid,pid,used_memory rows and 1 in the pid,used_memory rows.
__lease_apps_split_own() {
  local row pid
  LEASE_APPS_OWN='' LEASE_APPS_FOREIGN=''
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    pid=$(printf '%s\n' "$row" | cut -d, -f"$2" | tr -d ' ')
    if lease_pid_is_ours "$pid"; then
      LEASE_APPS_OWN+="${LEASE_APPS_OWN:+$'\n'}$row"
    else
      LEASE_APPS_FOREIGN+="${LEASE_APPS_FOREIGN:+$'\n'}$row"
    fi
  done <<< "$1"
}
# __lease_apps_own_line <rows> <pid field> <where>: one line naming the runner's own process(es) a
# guard did not count, each pid once with its exe (one process spans both cards as one pid), so the
# log still shows what was on the card.
__lease_apps_own_line() {
  local row pid exe said='' list=''
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    pid=$(printf '%s\n' "$row" | cut -d, -f"$2" | tr -d ' ')
    case ",$said," in
      *",$pid,"*) continue ;;
    esac
    said+="${said:+,}$pid"
    exe=$(readlink "/proc/$pid/exe" 2> /dev/null) || exe='?'
    list+="${list:+, }pid $pid (${exe:-?})"
  done <<< "$1"
  [ -n "$list" ] || return 0
  echo "[cards-own] $(now) the runner's own process(es) on $3, not counted: $list"
}

# __lease_gpu_holders <uuid> <compute-apps csv>: one line per process nvidia-smi listed — its pid,
# its exe and cwd, its age in seconds, its card memory — what a waiter, or the log of a run that
# ended beside the process, needs to name the co-tenant.
__lease_gpu_holders() {
  local row pid mem etime exe cwd
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    pid=${row%%,*} mem=${row#*,}
    pid=${pid// /}
    etime=$(ps -o etimes= -p "$pid" 2> /dev/null | tr -d ' ') || true
    exe=$(readlink "/proc/$pid/exe" 2> /dev/null) || true
    cwd=$(readlink "/proc/$pid/cwd" 2> /dev/null) || true
    echo "[lease]   pid $pid etime=${etime:-?} s exe=${exe:-?} cwd=${cwd:-?} mem=${mem# }"
  done <<< "$2"
}

lease_gpu_idle() {
  local uuid waited reported pids v
  [ -n "${TIMING_GPU:-}" ] || return 0
  for v in LEASE_GPU_WAIT LEASE_GPU_POLL LEASE_GPU_REPORT; do
    case ${!v} in
      '' | *[!0-9]* | 0)
        echo "[lease] $v is whole seconds above 0, got '${!v}'" >&2
        exit 64
        ;;
    esac
  done
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "[lease] nvidia-smi is not on PATH: the timing card's compute processes cannot be read, so no timed run starts beside them (exit 69)" >&2
    exit 69
  fi
  for uuid in $TIMING_GPU ${TIMING_GPU2:-}; do
    waited=0 reported=-1
    while :; do
      # Bounded (__witness_smi): a card off the bus can hang nvidia-smi, and this loop is inside
      # the lease.
      __witness_smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$uuid"
      if [ "$__witness_rc" != 0 ]; then
        echo "[lease] nvidia-smi on the timing card ($uuid) failed (rc $__witness_rc): its compute processes cannot be read, so no timed run starts beside them (exit 69)" >&2
        exit 69
      fi
      # The runner's own rows are not co-tenants (lease_pid_is_ours) — the rule the guards between
      # arms use; at the take no child of the runner exists yet, so this drops nothing here.
      __lease_apps_split_own "$__witness_out" 1
      [ -z "$LEASE_APPS_OWN" ] || __lease_apps_own_line "$LEASE_APPS_OWN" 1 'the timing card' >&2
      __witness_out=$LEASE_APPS_FOREIGN
      if [ -z "$__witness_out" ]; then
        if [ "$waited" = 0 ]; then
          echo "[lease] timing card idle: $uuid"
        else
          echo "[lease] timing card idle after ${waited} s: $uuid"
        fi
        break
      fi
      if [ "$reported" -lt 0 ] || [ $((waited - reported)) -ge "$LEASE_GPU_REPORT" ]; then
        echo "[lease] compute process(es) on the timing card ($uuid); waiting up to ${LEASE_GPU_WAIT} s, the lease held while it waits:"
        __lease_gpu_holders "$uuid" "$__witness_out"
        reported=$waited
      fi
      if [ "$waited" -lt "$LEASE_GPU_WAIT" ]; then
        sleep "$LEASE_GPU_POLL"
        waited=$((waited + LEASE_GPU_POLL))
      else
        pids=$(printf '%s\n' "$__witness_out" | cut -d, -f1 | tr -d ' ' | tr '\n' ' ')
        echo "[lease] the timing card ($uuid) still runs compute process(es) after ${LEASE_GPU_WAIT} s (pid $pids): contention, the run ends (exit 75)" >&2
        if [ "$reported" != "$waited" ]; then
          __lease_gpu_holders "$uuid" "$__witness_out"
        fi
        exit 75
      fi
    done
  done
}
# The lease's trace: one line per event in a file that outlives every runner, so a sitting can be
# read back with its owner, card and pid. /var/log/netdata-lease-gate.log stays the timing source
# (take and free at 0.5-2 s); this file adds who. The file is BLOOMERY_LEASE_TRACE, else
# /root/bloomery-sittings.log under the machine lease; under any other lock file (the stub tests) no
# trace is written unless BLOOMERY_LEASE_TRACE names one. A line that cannot be written is one named
# `[lease] trace not written` line on stderr and the lease goes on, as the timing-card record does.
#   take    lease_take, once the lease is held: its id (<epoch>.<pid>), the runner, the owner (the
#           tree's directory), the queue job when BLOOMERY_Q_JOB is set, the timing card(s), the card file
#   free    when the lease file turns free: a detached waiter probes it every 0.5 s (never a blocking
#           shared flock, which would fail the next take's `flock -n` for the probe's ms), so a runner
#           that died with a child holding descriptor 9 is still paired; held_s runs from the take to
#           the probe
#   free ... by=next-take   written by the next take when the last take has no free: the waiter died,
#           or it had not probed yet when the lease was free for less than a probe interval between
#           two takes; held_s is an upper bound
#   let_go  tools/ref/lease-hold.sh's exit: its descriptor is closed, the lease may be held on by
#           what the command left running (still_held=1; `?` when the probe could not tell)
LEASE_TRACE_ID=
LEASE_TRACE_FILE=
# __lease_trace_file: the trace's path for this lease, or nothing.
__lease_trace_file() {
  if [ -n "${BLOOMERY_LEASE_TRACE:-}" ]; then
    printf '%s' "$BLOOMERY_LEASE_TRACE"
  elif [ "$LEASE_LOCK" = /root/bloomery-cpu.lock ]; then
    printf '%s' /root/bloomery-sittings.log
  fi
}
# __lease_trace_header: the lines that open a new trace file.
__lease_trace_header() {
  cat << 'EOF'
# The machine lease's trace (tools/ref/lease.sh). /var/log/netdata-lease-gate.log stays the timing source
# (take and free at 0.5-2 s); this file adds the owner, the card and the pid.
# <utc> take id=<epoch>.<pid> epoch=<s> pid=<runner pid> owner=<tree> [job=<queue job>] runner=<script> timing_gpu=<uuid[,uuid]|none> card_file=<card>
# <utc> free id=<id> epoch=<s> held_s=<s> by=waiter|next-take   (next-take: no waiter line came first, the waiter died or the lease was free for less than one 0.5 s probe; held_s is an upper bound)
# <utc> let_go id=<id> still_held=0|1|?   (tools/ref/lease-hold.sh exited; the lease may be held on by what it left running)
EOF
}
# __lease_trace_write <file> <lines>: the lines appended in one write; one named line on stderr and 1
# when the write fails.
__lease_trace_write() {
  local err
  if err=$({ printf '%s' "$2" >> "$1"; } 2>&1); then return 0; fi
  echo "[lease] trace not written: ${err##*: } ($1); the lease goes on" >&2
  return 1
}
# __lease_trace_waiter <lease file> <trace file> <id> <take epoch>: runs detached (lease_trace_take):
# probes the lease file every 0.5 s and, when it is free, appends the `free` line of <id>, unless the
# next take already closed it (a probe can miss a short gap between two runs of a sitting). It ends
# without a line on a lease file it cannot test 20 probes in a row: the next take pairs the line.
__lease_trace_waiter() {
  local rc bad=0 e
  while :; do
    rc=0
    flock -s -n -E 75 "$1" true 2> /dev/null || rc=$?
    case $rc in
      0) break ;;
      75) bad=0 ;;
      *)
        bad=$((bad + 1))
        [ "$bad" -lt 20 ] || return 1
        ;;
    esac
    sleep 0.5
  done
  e=$(date +%s)
  ! grep -q " free id=$3 " "$2" 2> /dev/null || return 0
  printf '%s free id=%s epoch=%s held_s=%s by=waiter\n' "$(now)" "$3" "$e" "$((e - $4))" >> "$2" 2> /dev/null
}
# lease_trace_take: the take's line, after the lease is held; the previous take's `free` first when it
# has none. Starts the waiter. Sets LEASE_TRACE_ID and LEASE_TRACE_FILE (empty when nothing was written).
lease_trace_take() {
  local f epoch id last lid lt payload line gpu
  LEASE_TRACE_ID='' LEASE_TRACE_FILE=''
  f=$(__lease_trace_file)
  [ -n "$f" ] || return 0
  epoch=$(date +%s)
  id=$epoch.$$
  gpu=${TIMING_GPU:-none}${TIMING_GPU2:+,$TIMING_GPU2}
  payload=''
  if [ -e "$f" ]; then
    last=$(grep -E '^[^#].* take id=' "$f" 2> /dev/null | tail -n 1) || true
    if [ -n "$last" ]; then
      lid=${last#* take id=}
      lid=${lid%% *}
      if ! grep -q " free id=$lid " "$f" 2> /dev/null; then
        lt=${lid%%.*}
        case $lt in '' | *[!0-9]*) lt=$epoch ;; esac
        payload="$(now) free id=$lid epoch=$epoch held_s=$((epoch - lt)) by=next-take"$'\n'
      fi
    fi
  else
    payload=$(__lease_trace_header)$'\n'
  fi
  line="$(now) take id=$id epoch=$epoch pid=$$ owner=${LEASE_TREE##*/}${BLOOMERY_Q_JOB:+ job=$BLOOMERY_Q_JOB} runner=${0##*/} timing_gpu=$gpu card_file=${BLOOMERY_LEASE_CARD:--}"
  __lease_trace_write "$f" "$payload${line:0:399}"$'\n' || return 0
  LEASE_TRACE_ID=$id LEASE_TRACE_FILE=$f
  if ! command -v setsid > /dev/null 2>&1; then
    echo "[lease] trace waiter not started: setsid is not on PATH, so the free line is left to the next take" >&2
    return 0
  fi
  # Detached: a subshell that exits at once, so no `wait` of the runner meets it; its own session; no
  # descriptor 9, or it would hold the lease it waits for; no stdin, stdout or stderr, or an ssh
  # session never ends; its directory is /, so it pins no track's directory while it waits.
  (setsid bash -c "$(declare -f now __lease_trace_waiter)"$'\n''cd / && __lease_trace_waiter "$@"' _ "$LEASE_LOCK" "$f" "$id" "$epoch" 9>&- < /dev/null > /dev/null 2>&1 &)
}
# lease_trace_let_go <0|1|?>: lease-hold.sh's `let_go` line for the lease this process took.
lease_trace_let_go() {
  [ -n "$LEASE_TRACE_ID" ] || return 0
  __lease_trace_write "$LEASE_TRACE_FILE" "$(now) let_go id=$LEASE_TRACE_ID still_held=$1"$'\n' || true
}

lease_take() {
  if [ -n "${BLOOMERY_LEASE_HELD:-}" ]; then
    echo "[lease] refused: this run is inside tools/ref/lease-hold.sh (pid $BLOOMERY_LEASE_HELD), which holds the lease; a second lease would wait on it" >&2
    exit 64
  fi
  # A GPU runner (timing-card.sh sourced: it defines witness_card) must name its timing card, or the
  # record below would say `none` and let a forced gate share that card.
  if declare -F witness_card > /dev/null && [ -z "${TIMING_GPU:-}" ]; then
    echo "[lease] refused: a GPU timing runner (timing-card.sh sourced) with TIMING_GPU empty — the lease would record no timing card${TIMING_CARDS_WHY:+ ($TIMING_CARDS_WHY)}" >&2
    exit 64
  fi
  if [ "$LEASE_LOCK" != /root/bloomery-cpu.lock ] && [ -e /root/bloomery-cpu.lock ]; then
    echo "[lease] refused: BLOOMERY_LEASE_LOCK=$LEASE_LOCK on the machine whose lease is /root/bloomery-cpu.lock: a run under another lock would share the box with a real sitting" >&2
    exit 64
  fi
  lease_card || exit $?
  [ "$LEASE_LOCK" = /root/bloomery-cpu.lock ] ||
    echo "[lease] BLOOMERY_LEASE_LOCK=$LEASE_LOCK is not the machine lease: nothing measured under it is admissible"
  # The runner the card guards compare against (lease_pid_is_ours): recorded once, so a helper
  # called from a subshell still compares against the runner.
  LEASE_OWNER_PID=$$
  exec 9>"$LEASE_LOCK"
  if ! flock -n 9; then
    local waited=0
    echo "[lease] $LEASE_LOCK is held; waiting up to 30 min. Its holder:"
    lease_holders "$LEASE_LOCK"
    until flock -w 60 9; do
      waited=$((waited + 1))
      if [ "$waited" -ge 30 ]; then
        {
          echo "[lease] timed out after 30 min; the holder at the timeout:"
          lease_holders "$LEASE_LOCK"
        } >&2
        exit 75
      fi
      echo "[lease] still held after $waited min by:"
      lease_holders "$LEASE_LOCK"
    done
  fi
  lease_gpu_idle
  # The timing-card record tools/gpu-gate.sh reads while the lease is held: this runner and its card (both
  # cards in the two-card mode, the A6000 first), `none` for one that times no GPU.
  { printf 'pid=%s timing_gpu=%s\n' "$$" "${TIMING_GPU:-none}${TIMING_GPU2:+,$TIMING_GPU2}" > "$LEASE_LOCK.card.$$" && mv -f "$LEASE_LOCK.card.$$" "$LEASE_LOCK.card"; } ||
    echo "[lease] the timing-card record $LEASE_LOCK.card was not written: forced GPU gates refuse (75) while this lease is held" >&2
  lease_trace_take
  echo "[lease] held by pid $$ at $(now)"
  lease_netdata
}

# lease_bounded <seconds> <command…>: the command under `timeout --kill-after=10 <seconds>`, its exit
# code returned. Every process a runner starts under the lease goes through it or its own `timeout`:
# the lease stays held while any process that inherited descriptor 9 lives, so a hung child must end
# at a bound instead of holding the machine. A command cut off there prints `[bound]` on stderr: a
# child that hangs is a failed run, not a slow one. The command is an executable, not a shell
# function or an assignment: `lease_bounded "$LEASE_ARM_BOUND" env K=V "$BIN" …`. LEASE_ARM_BOUND is
# BLOOMERY_ARM_BOUND, default 900 s — the arm bound the depth, nsys and ncu runners read too; a
# runner whose children run longer names its own. A bound that is not a positive integer is refused
# (rc 64) before anything runs.
# shellcheck disable=SC2034 # read by the runners that source this file
LEASE_ARM_BOUND=${BLOOMERY_ARM_BOUND:-900}
lease_bounded() {
  local bound=${1:-} rc=0
  case $bound in
    '' | *[!0-9]* | 0*)
      echo "[bound] lease_bounded: the bound is a positive number of seconds, got '$bound' (BLOOMERY_ARM_BOUND?)" >&2
      return 64
      ;;
  esac
  shift
  timeout --kill-after=10 "$bound" "$@" || rc=$?
  case $rc in
    124) echo "[bound] $1 cut off at its ${bound} s bound (rc 124)" >&2 ;;
    137) echo "[bound] $1 killed (rc 137: the bound's KILL ${bound} s + 10 s in, or another SIGKILL)" >&2 ;;
  esac
  return "$rc"
}

# netdata-lease-gate.service (rig-log configs/) freezes netdata while this lease is held; it polls,
# so wait up to 2 s for the freeze and print the state. `running` on this line means the run was
# measured with netdata's collectors live (its GPU collector queries both cards every 2 s).
lease_netdata() {
  local i s
  if ! systemctl is-active --quiet netdata.service 2> /dev/null; then
    echo "[lease] netdata: not active"
    return 0
  fi
  for ((i = 0; i < 20; i++)); do
    s=$(systemctl show -p FreezerState --value netdata.service 2> /dev/null || true)
    [ "$s" = frozen ] && break
    sleep 0.1
  done
  echo "[lease] netdata: ${s:-unknown}"
}

lease_release() { exec 9>&-; }

# lease_card: card.py's check of BLOOMERY_LEASE_CARD and its [lease] lines; returns card.py's code.
lease_card() {
  local rounds=${ROUNDS:-${BLOOMERY_AB_ROUNDS:-}}
  if [ -z "$LEASE_TREE" ]; then
    echo "[lease] refused: the tree of tools/ref/lease.sh is not found, so neither is its card" >&2
    return 66
  fi
  python3 "$LEASE_TREE/tools/ref/card.py" lease "${BLOOMERY_LEASE_CARD:-}" \
    ${rounds:+--rounds "$rounds"} ${ROUND_MINUTES:+--round-minutes "$ROUND_MINUTES"}
}

# CPU contention between arms. The lease serializes timed runs, not builds: another round's cargo
# build, a CUDA C++ build (nvcc drives cicc and ptxas, which carry its long passes — build-ref, the
# mistral.rs build), or a reference engine this runner did not start, shares the cores a timed arm
# runs on, and the `busiest` witness field only shows it. guard_cpu reads the cpu time of the
# processes whose name is in CPU_BUSY_COMMS (BLOOMERY_CPU_BUSY_COMMS) over the interval since its
# last reading — utime + stime from /proc/<pid>/stat (BLOOMERY_LEASE_PROC names another tree, for the
# stub tests), a process that started inside the interval counted whole — and leaves out the
# runner's own: every process under the runner's pid ($$), such as the load process an arm list
# keeps between its arms. Above CPU_BUSY_PCT (BLOOMERY_CPU_BUSY_PCT, percent of one cpu — a chosen
# threshold, not a measured one) it prints `[cpu-busy]` on stderr and sets CPU_BUSY_TAG to
# ` [cpu-busy]` for the runner's row; with BLOOMERY_OTHER_STRICT=1 it prints a witness block and
# exits 75 instead, as guard_other does for a busy other card. A reading spans at least a second: a
# call less than a second after the last reading only starts the next interval (a pre-arm guard right
# after the previous arm's post guard: the arm's own interval is then the post guard's), and the
# first call waits out a second when a named process is running. A process that exits inside an
# interval is not counted. The runner clears CPU_BUSY_TAG when a row starts.
CPU_BUSY_COMMS=${BLOOMERY_CPU_BUSY_COMMS:-cargo rustc cc1plus nvcc cicc ptxas llama-bench generate_ds41}
CPU_BUSY_PCT=${BLOOMERY_CPU_BUSY_PCT:-50}
CPU_BUSY_TAG=
# The last reading: the uptime and every process's (pid, start, cpu ticks) then (CPU_BUSY_SNAP); the
# named processes that are not the runner's, `<sum of %CPU> <name %CPU;...>` (CPU_BUSY_READING), the
# runner's own it left out (CPU_BUSY_OWN), the interval in seconds (CPU_BUSY_SPAN, empty when the
# call only started one), and why no reading was taken (CPU_BUSY_ERR).
CPU_BUSY_SNAP='' CPU_BUSY_READING='' CPU_BUSY_OWN='' CPU_BUSY_SPAN='' CPU_BUSY_ERR=''

# cpu_busy_sample: a reading over the interval since the last one, into the variables above; with no
# earlier reading, over the next second (at once when no named process runs). Returns 1 with
# CPU_BUSY_ERR set when the process tree cannot be read.
cpu_busy_sample() {
  local out
  # The last reading goes in on stdin: a few hundred processes' lines can outgrow one argument.
  out=$(python3 -c "$(cpu_busy_py)" "${BLOOMERY_LEASE_PROC:-/proc}" "$$" "$CPU_BUSY_COMMS" <<< "$CPU_BUSY_SNAP") \
    || { CPU_BUSY_ERR="python3 failed reading ${BLOOMERY_LEASE_PROC:-/proc}"; return 1; }
  eval "$out"
  [ -z "$CPU_BUSY_ERR" ]
}

# cpu_busy_py: cpu_busy_sample's program — argv the tree, the runner's pid and the names, stdin the
# last reading; out, the variables as shell assignments.
cpu_busy_py() {
  cat << 'PY'
import os
import shlex
import sys
import time

proc, root, comms = sys.argv[1], int(sys.argv[2]), set(sys.argv[3].split())
snap = sys.stdin.read().strip()
MIN_SPAN = 1.0
hz = os.sysconf('SC_CLK_TCK')


def emit(**kv):
    for k, v in kv.items():
        print(f'CPU_BUSY_{k}={shlex.quote(v)}')
    sys.exit(0)


def now():
    """The uptime, and pid -> (comm, ppid, start, ticks) of every process."""
    try:
        with open(os.path.join(proc, 'uptime')) as f:
            up = float(f.read().split()[0])
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except (OSError, ValueError, IndexError) as e:
        emit(ERR=f'{proc} cannot be read ({e})')
    ps = {}
    for d in pids:
        try:
            with open(os.path.join(proc, d, 'stat')) as f:
                s = f.read()
        except OSError:
            continue
        lo, hi = s.find('('), s.rfind(')')
        f = s[hi + 2:].split()
        if lo < 0 or hi < lo or len(f) < 20:
            emit(ERR=f'{proc}/{d}/stat is not a stat line: {s[:80]!r}')
        ps[int(d)] = (s[lo + 1:hi], int(f[1]), int(f[19]), int(f[11]) + int(f[12]))
    return up, ps


def text(up, ps):
    return '\n'.join([repr(up)] + [f'{p} {v[2]} {v[3]}' for p, v in ps.items()])


cur = now()
if not snap:
    if not any(v[0] in comms for v in cur[1].values()):
        emit(SNAP=text(*cur), SPAN='', READING='0.0 ', OWN='', ERR='')
    prev = cur
    time.sleep(MIN_SPAN)
    cur = now()
else:
    lines = snap.split('\n')
    prev = (float(lines[0]), {int(p): (None, None, int(st), int(t))
                              for p, st, t in (x.split() for x in lines[1:])})
span = cur[0] - prev[0]
if span < MIN_SPAN and snap:
    emit(SNAP=text(*cur), SPAN='', READING='0.0 ', OWN='', ERR='')
if span <= 0:
    emit(ERR=f'an interval of {span:.2f} s: the uptime in {proc} did not move')


def own(pid):
    seen = 0
    while pid in cur[1] and seen < 256:
        if pid == root:
            return True
        pid, seen = cur[1][pid][1], seen + 1
    return False


mine, theirs, total = [], [], 0.0
for pid, (comm, _, start, ticks) in sorted(cur[1].items()):
    if comm not in comms:
        continue
    was = prev[1].get(pid)
    base = was[3] if was is not None and was[2] == start else 0
    if ticks < base:
        emit(ERR=f'pid {pid} ({comm}): cpu time {ticks} ticks, {base} at the interval start')
    pct = (ticks - base) / hz / span * 100
    if own(pid):
        mine.append(f'{comm} {pct:.1f}%;')
    else:
        theirs.append(f'{comm} {pct:.1f}%;')
        total += pct
emit(SNAP=text(*cur), SPAN=f'{span:.1f}', READING=f'{total:.1f} ' + ''.join(theirs), OWN=''.join(mine), ERR='')
PY
}

# cpu_busy_reading: `<sum of %CPU> <name %CPU;...>` over the named processes that are not the
# runner's (cpu_busy_sample), or `no reading: <why>`.
cpu_busy_reading() {
  if cpu_busy_sample; then echo "$CPU_BUSY_READING"; else echo "no reading: $CPU_BUSY_ERR"; fi
}

guard_cpu() {
  local tag=$1 sum
  case $CPU_BUSY_PCT in
    '' | *[!0-9.]* | *.*.*)
      echo "guard_cpu: BLOOMERY_CPU_BUSY_PCT is a percentage, got '$CPU_BUSY_PCT'" >&2
      exit 64
      ;;
  esac
  if ! cpu_busy_sample; then
    echo "guard_cpu: $tag: no cpu reading: $CPU_BUSY_ERR" >&2
    exit 70
  fi
  sum=${CPU_BUSY_READING%% *}
  awk -v s="$sum" -v t="$CPU_BUSY_PCT" 'BEGIN { exit !(s > t) }' || return 0
  echo "[cpu-busy] $(now) $tag: ${sum}% > ${CPU_BUSY_PCT}% of one cpu over ${CPU_BUSY_SPAN} s [${CPU_BUSY_READING#* }]${CPU_BUSY_OWN:+ (not counted, under the runner: $CPU_BUSY_OWN)}" >&2
  # shellcheck disable=SC2034 # the sourcing runner reads it into its row
  CPU_BUSY_TAG=' [cpu-busy]'
  if [ "${BLOOMERY_OTHER_STRICT:-}" = 1 ]; then
    witness abort-cpu >&2
    exit 75
  fi
  return 0
}

witness() {
  local tag=$1 field fn
  __witness_indent=
  if [ -z "${WITNESS[*]+set}" ]; then
    echo "witness: this runner lists no WITNESS fields" >&2
    return 64
  fi
  for field in "${WITNESS[@]}"; do
    if [ "$field" = indent ]; then
      __witness_indent='    '
      continue
    fi
    fn=__witness_${field//-/_}
    if ! declare -F "$fn" > /dev/null; then
      echo "witness: no field '$field' (tools/ref/lease.sh lists them)" >&2
      return 64
    fi
    "$fn" "$tag"
  done
}

__witness_head() { echo "--- witness $1 $(now) ---"; }
__witness_head_open() { echo "--- witness $1 $(now)"; }
__witness_head_epoch() { echo "--- witness $1 $(now) epoch $(date +%s) ---"; }
# The fields that ask the cards. nvidia-smi fails when a card falls off the bus (Xid 79), and a field
# must then say so — `<field>: unavailable (rc N)` — rather than print nothing, print an empty list
# that reads as "no process", or end a runner under `set -e -o pipefail` before its witness is out.
# __witness_smi <nvidia-smi arguments…> puts the output in __witness_out and the exit code in
# __witness_rc.
# A card that falls off the bus can also hang nvidia-smi: where `timeout` exists (the box) the query is
# cut off after 30 s and the field reads `unavailable (rc 124)`.
__witness_smi() {
  __witness_rc=0
  if command -v timeout > /dev/null; then
    __witness_out=$(timeout --kill-after=5 30 nvidia-smi "$@") || __witness_rc=$?
  else
    __witness_out=$(nvidia-smi "$@") || __witness_rc=$?
  fi
}
# shellcheck disable=SC2001 # a prefix on every line of a multi-line value: sed, not ${var//}
__witness_lines() { [ -z "$__witness_out" ] || sed "s/^/${__witness_indent}/" <<< "$__witness_out"; }
__witness_joined() { [ -z "$__witness_out" ] || printf '%s\n' "$__witness_out" | tr '\n' "$1"; }
__witness_head_load() {
  local gpu
  __witness_smi --query-gpu=utilization.gpu --format=csv,noheader
  if [ "$__witness_rc" = 0 ]; then gpu=$(__witness_joined ' '); else gpu="unavailable (rc $__witness_rc)"; fi
  echo "--- witness $1 $(now) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) gpu=$gpu"
}
# __lease_timing_labels: the `card` field's apps-line labels of the cards this run times (`a6000`,
# `3090`) — TIMING_GPU and TIMING_GPU2 against the UUIDs tools/ref/cards.sh resolved for whoever
# sourced this file; empty when no timed card is one of them (its apps lines stay untagged; a card
# cards.sh could not name is named in those lines themselves).
__lease_timing_labels() {
  local uuid out=''
  for uuid in ${TIMING_GPU:-} ${TIMING_GPU2:-}; do
    if [ -n "${GPU_A6000:-}" ] && [ "$uuid" = "$GPU_A6000" ]; then out="$out a6000"; fi
    if [ -n "${GPU_3090:-}" ] && [ "$uuid" = "$GPU_3090" ]; then out="$out 3090"; fi
  done
  printf '%s' "$out"
}
# The `card` field's lines are timing-card.sh's witness_card; the tag is the lease's view of them.
# An apps line of a card this run times that lists a process is a witness block taken beside a
# co-tenant (the take refused one in lease_gpu_idle; guard_other and guard_cards police the arms),
# so those lines end with ` [timing-card-busy]` and one log grep finds the runs whose numbers were
# measured beside one. The regex rides in a variable: in [[ =~ ]] an unquoted \[ would lose its
# backslash to quote removal and open a bracket expression.
__witness_card() {
  local line label re out labels
  out=$(witness_card)
  labels=$(__lease_timing_labels)
  if [ -z "$out" ] || [ -z "$labels" ]; then
    [ -z "$out" ] || printf '%s\n' "$out"
    return 0
  fi
  while IFS= read -r line; do
    for label in $labels; do
      re="^( *${label}-apps): \[(.+)\]$"
      if [[ $line =~ $re ]]; then
        if [ -n "${BASH_REMATCH[2]}" ]; then
          line="$line [timing-card-busy]"
        fi
        break
      fi
    done
    printf '%s\n' "$line"
  done <<< "$out"
}
__witness_loadavg() { echo "${__witness_indent}loadavg: $(cat /proc/loadavg)"; }
__witness_pressure_cpu() { echo "${__witness_indent}pressure-cpu: $(grep '^some' /proc/pressure/cpu | head -n1)"; }
__witness_pressure_io() { echo "${__witness_indent}pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"; }
__witness_pressure_io_avg10() { echo "${__witness_indent}pressure-io avg10: $(grep '^some' /proc/pressure/io | head -n 1)"; }
__witness_gpus() {
  __witness_smi --query-gpu=index,name,utilization.gpu,power.draw --format=csv,noheader
  if [ "$__witness_rc" = 0 ]; then __witness_lines; else echo "${__witness_indent}gpus: unavailable (rc $__witness_rc)"; fi
}
__witness_gpu_apps() {
  __witness_smi --query-compute-apps=pid,used_memory --format=csv,noheader
  if [ "$__witness_rc" = 0 ]; then
    echo "${__witness_indent}gpu-apps: [$(__witness_joined ';')]"
  else
    echo "${__witness_indent}gpu-apps: unavailable (rc $__witness_rc)"
  fi
}
__witness_stage0_gpus() {
  __witness_smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw --format=csv
  if [ "$__witness_rc" = 0 ]; then __witness_lines; else echo "${__witness_indent}stage0-gpus: unavailable (rc $__witness_rc)"; fi
}
__witness_stage0_apps() {
  # The 3090's UUID comes from tools/ref/cards.sh through whoever sourced this file; a card it
  # could not name is named here, not queried with an empty -i (an empty list would read as
  # "no process").
  if [ -z "${GPU_3090:-}" ]; then
    echo "${__witness_indent}compute-apps-3090: unresolved (no 3090 UUID${CARDS_ERROR:+: $CARDS_ERROR})"
    return 0
  fi
  __witness_smi --query-compute-apps=pid,used_memory --format=csv -i "$GPU_3090"
  if [ "$__witness_rc" = 0 ]; then
    echo "${__witness_indent}compute-apps-3090:"
    __witness_lines
  else
    echo "${__witness_indent}compute-apps-3090: unavailable (rc $__witness_rc)"
  fi
}
__witness_lock_holder() { echo "${__witness_indent}lock-holder-pid: $$"; }
__witness_model() { echo "${__witness_indent}model: ${MODEL_NAME:-?}"; }
__witness_busiest() {
  echo "${__witness_indent}busiest: $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -n 4 | awk '{printf "%s %s%% | ", $1, $2}')"
}
__witness_threads() {
  echo "${__witness_indent}threads: BLOOMERY_THREADS=${BLOOMERY_THREADS:-<default>} spin=${BLOOMERY_SPIN:-<default>}"
}
__witness_core() {
  echo "${__witness_indent}core: $CORE ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'))"
}
__witness_core_mhz() {
  echo "${__witness_indent}cpu-mhz: $(awk -v c="$CORE" '$1 == "processor" { p = $3 } $1 == "cpu" && $2 == "MHz" && p == c { print $4; exit }' /proc/cpuinfo)"
}
__witness_cpu() { echo "${__witness_indent}cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"; }
__witness_cpu_mhz_range() {
  echo "${__witness_indent}cpu-mhz min/max: $(awk '$1 == "cpu" && $2 == "MHz" { if (lo == "" || $4 < lo) lo = $4; if ($4 > hi) hi = $4 } END { print lo, hi }' /proc/cpuinfo)"
}
# cpu_freq_summary: `MHz mean <m> min <lo> max <hi> over <n> cpus governor=<g>` from every cpu's
# cpufreq scaling_cur_freq (kHz), read by one awk; `unavailable (…)` where the files are missing.
cpu_freq_summary() {
  local gov='?' g=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  local -a files=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq)
  if [ ! -r "${files[0]}" ]; then
    echo "unavailable (no cpufreq scaling_cur_freq under /sys/devices/system/cpu)"
    return 0
  fi
  [ -r "$g" ] && gov=$(< "$g")
  awk -v gov="$gov" '{ s += $1; n++; if (n == 1 || $1 < lo) lo = $1; if ($1 > hi) hi = $1 }
    END { printf "MHz mean %.0f min %.0f max %.0f over %d cpus governor=%s\n", s / n / 1000, lo / 1000, hi / 1000, n, gov }' "${files[@]}"
}
__witness_cpu_freq() { echo "${__witness_indent}cpu-freq: $(cpu_freq_summary)"; }
__witness_meminfo() {
  echo "${__witness_indent}meminfo cached/free kB: $(awk '/^Cached:/{c=$2} /^MemFree:/{f=$2} END{print c, f}' /proc/meminfo)"
}
__witness_mem() {
  echo "${__witness_indent}mem: $(grep -E '^(MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
}
__witness_pgmajfault() { echo "${__witness_indent}pgmajfault: $(awk '$1 == "pgmajfault" {print $2}' /proc/vmstat)"; }
__witness_read_sectors() {
  local dev sectors
  dev=$(df --output=source "$MODEL" 2>/dev/null | tail -n 1) || true
  sectors=$(awk '{print $3}' "/sys/class/block/${dev#/dev/}/stat" 2>/dev/null || echo '?')
  echo "${__witness_indent}read-sectors: $sectors ($dev, 512 B each)"
}
__witness_table() { echo "${__witness_indent}table: $DIR on $SRC (block device $DEV)"; }
# Fields 1 and 3 of /sys/block/<dev>/stat: completed read I/Os and sectors read (512 B).
__witness_blockstat() { echo "${__witness_indent}blockstat($DEV) read_ios/read_sectors: $(awk '{print $1, $3}' "$STAT")"; }
__witness_binary() { echo "${__witness_indent}binary: $BIN sha256=$BIN_SHA"; }

# The prompt of the depth tables, n ids: BOS (100000), then an LCG walk over ids [1000, 91000).
# deepseek2's ids. The CPU and GPU depth runners and the two counter runners feed this one sequence,
# so their rows are about the same prompt. awk computes it in doubles, so past the second id it is
# not the exact 64-bit LCG that gate_e2e's lcg_prompt computes: the tables' prompt is this one.
lcg_prompt() {
  awk -v n="$1" 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}'
}

# The bloomery-decode binary: DECODE_BIN in the tree the runner stands in (tools/box.sh runs every
# command from the track's own directory), `decode_bin <tree>` in a sibling tree under ~/repo on the box.
DECODE_BIN=target/release/bloomery-decode
decode_bin() { echo "$HOME/repo/$1/$DECODE_BIN"; }

# `bash tools/ref/lease.sh --self-test`: lease_gpu_idle and the witness tag, on a stub nvidia-smi
# first on PATH in a temp dir (the made-up UUIDs the stub tests use, never the box's), with no
# card, no lock and no /proc — the holder lines name a pid that does not resolve. The waits scale
# to seconds (LEASE_GPU_POLL/REPORT/WAIT of 1/1/2), and the four lease_gpu_idle cases sleep 4 s
# between them; the trace cases (a temp lease and trace, no /root) follow. just check-recipes runs this
# on the Mac.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0
  check() {
    if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi
  }
  matches() { # <name> <ERE> <text>: the text has a line matching the ERE
    if printf '%s\n' "$3" | grep -Eq -- "$2"; then echo "ok $1"; else echo "FAIL $1: no line matches /$2/ in [$3]"; fails=$((fails + 1)); fi
  }
  t=$(mktemp -d "${TMPDIR:-/tmp}/lease-self-test.XXXXXX") || exit 70
  trap 'rm -rf "$t"' EXIT
  mkdir "$t/bin" "$t/nothing"
  # The stub nvidia-smi: rc $STUB_RC when set; its call count in $t/calls; the compute apps of
  # $STUB_APPS, suppressed once the count reaches $STUB_VANISH_AT.
  cat > "$t/bin/nvidia-smi" << 'EOF'
#!/bin/sh
n=$(cat "$STUB_DIR/calls" 2> /dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$STUB_DIR/calls"
[ "${STUB_RC:-0}" = 0 ] || exit "$STUB_RC"
if [ -n "${STUB_VANISH_AT:-}" ] && [ "$n" -ge "$STUB_VANISH_AT" ]; then exit 0; fi
[ -z "${STUB_APPS:-}" ] || printf '%s\n' "$STUB_APPS"
exit 0
EOF
  chmod +x "$t/bin/nvidia-smi"
  A6000=GPU-00000000-0000-0000-0000-000000000000
  # idle <K=V…>: lease_gpu_idle in a subshell (its refusal exits stay there) with the stub first on
  # PATH, TIMING_GPU=$A6000 and the waits scaled; its output in OUT, its rc in RC.
  idle() {
    local k
    RC=0
    OUT=$(
      export STUB_DIR="$t"
      for k in "$@"; do export "$k"; done
      export PATH="$t/bin:$PATH" TIMING_GPU="$A6000" LEASE_GPU_POLL=1 LEASE_GPU_REPORT=1 LEASE_GPU_WAIT=2
      lease_gpu_idle 2>&1
    ) || RC=$?
  }
  idle
  check a-idle-rc "$RC" 0
  matches a-idle-line "^\[lease\] timing card idle: $A6000\$" "$OUT"
  idle 'STUB_APPS=4242, 100 MiB' 'STUB_VANISH_AT=3'
  check b-vanish-rc "$RC" 0
  matches b-idle-after "^\[lease\] timing card idle after [0-9]+ s: $A6000\$" "$OUT"
  matches b-holder-named '^\[lease\]   pid 4242 etime=' "$OUT"
  idle 'STUB_APPS=4242, 100 MiB'
  check c-stay-rc "$RC" 75
  matches c-final-line "^\[lease\] the timing card \($A6000\) still runs compute process\(es\) after 2 s \(pid 4242 \): contention, the run ends \(exit 75\)" "$OUT"
  matches c-holder-named '^\[lease\]   pid 4242 etime=' "$OUT"
  idle 'STUB_RC=9'
  check d-smi-fail-rc "$RC" 69
  matches d-smi-fail-line "nvidia-smi on the timing card \($A6000\) failed \(rc 9\)" "$OUT"
  # No nvidia-smi anywhere (an empty PATH: the branch runs builtins only — command -v, echo, exit).
  RC=0
  OUT=$(export PATH="$t/nothing" TIMING_GPU="$A6000"; lease_gpu_idle 2>&1) || RC=$?
  check e-no-smi-rc "$RC" 69
  matches e-no-smi-line '^\[lease\] nvidia-smi is not on PATH' "$OUT"
  # The witness tag: a busy apps line of a timed card ends with ` [timing-card-busy]`, an empty
  # one and the other card's stay as they are, both cards carry it in the two-card mode, and a
  # runner that times no card tags nothing.
  GPU_A6000=$A6000 GPU_3090=GPU-11111111-1111-1111-1111-111111111111
  witness_card() {
    printf '    3090-apps: [%s]\n' "${W3090_APPS:-}"
    printf '    a6000-apps: [%s]\n' "${WA6000_APPS:-}"
  }
  TIMING_GPU=$A6000 W3090_APPS='' WA6000_APPS='4242, 100 MiB;'
  RC=0
  OUT=$(__witness_card 2>&1) || RC=$?
  check tag-rc "$RC" 0
  matches tag-busy '^    a6000-apps: \[4242, 100 MiB;\] \[timing-card-busy\]$' "$OUT"
  matches tag-other-card-untouched '^    3090-apps: \[\]$' "$OUT"
  WA6000_APPS=''
  OUT=$(__witness_card 2>&1)
  matches tag-empty-untouched '^    a6000-apps: \[\]$' "$OUT"
  WA6000_APPS='4242, 100 MiB;' W3090_APPS='4243, 64 MiB;' TIMING_GPU2=$GPU_3090
  OUT=$(__witness_card 2>&1)
  matches tag-twocard-a6000 '^    a6000-apps: \[4242, 100 MiB;\] \[timing-card-busy\]$' "$OUT"
  matches tag-twocard-3090 '^    3090-apps: \[4243, 64 MiB;\] \[timing-card-busy\]$' "$OUT"
  unset TIMING_GPU2
  TIMING_GPU='' WA6000_APPS='4242, 100 MiB;'
  OUT=$(__witness_card 2>&1)
  matches tag-no-timing-passthrough '^    a6000-apps: \[4242, 100 MiB;\]$' "$OUT"
  TIMING_GPU=$A6000
  witness_card() { echo '    a6000-apps: unresolved (no A6000 UUID)'; }
  OUT=$(__witness_card 2>&1)
  matches tag-unresolved-untouched '^    a6000-apps: unresolved \(no A6000 UUID\)$' "$OUT"
  # ---- lease_pid_is_ours ----
  # A made-up /proc tree (BLOOMERY_LEASE_PROC, as the card tests use it: this runs on the Mac too):
  # pid 101 holds the lease's descriptor 9 as the runner does, 102 descends from it without the
  # descriptor, 103 is foreign under init, and nothing holds 4000000. The runner is this shell,
  # which opens the lease on its own descriptor 9.
  proc=$t/proc
  mkdir -p "$proc/$$/fd" "$proc/101/fd" "$proc/102" "$proc/103"
  exec 9>"$t/lease"
  ln -s "$t/lease" "$proc/$$/fd/9"
  ln -s "$t/lease" "$proc/101/fd/9"
  printf '101 (own child) S %s 1 0 0 0 0\n' "$$" > "$proc/101/stat"
  printf '102 (own grandchild) S 101 1 0 0 0 0\n' > "$proc/102/stat"
  printf '103 (foreign) S 1 1 0 0 0 0\n' > "$proc/103/stat"
  LEASE_OWNER_PID=$$ BLOOMERY_LEASE_PROC=$proc
  # A numeric pid is answered in silence: a guard asks per row and per poll, so the not-ours path
  # prints nothing (the named line is the non-numeric pid's alone).
  RC=0 OUT=$(lease_pid_is_ours 101 2>&1) || RC=$?
  check own-fd9-rc "$RC" 0
  check own-fd9-silent "$OUT" ''
  RC=0 OUT=$(lease_pid_is_ours 102 2>&1) || RC=$?
  check own-descend-rc "$RC" 0
  RC=0 OUT=$(lease_pid_is_ours 103 2>&1) || RC=$?
  check foreign-rc "$RC" 1
  check foreign-silent "$OUT" ''
  RC=0 OUT=$(lease_pid_is_ours 4000000 2>&1) || RC=$?
  check gone-rc "$RC" 1
  check gone-silent "$OUT" ''
  # The runner itself (a runner that execs into its engine keeps LEASE_OWNER_PID's pid) is ours.
  RC=0 OUT=$(lease_pid_is_ours $$ 2>&1) || RC=$?
  check own-self-rc "$RC" 0
  RC=0
  OUT=$(lease_pid_is_ours not-a-pid 2>&1) || RC=$?
  check nonnumeric-rc "$RC" 1
  matches nonnumeric-line "^\[lease\] lease_pid_is_ours: a pid is a positive number, got 'not-a-pid': it cannot be proven ours, so it counts as busy\$" "$OUT"
  unset LEASE_OWNER_PID BLOOMERY_LEASE_PROC
  exec 9>&-
  # ---- end lease_pid_is_ours ----
  # ---- the trace ----
  # Real flock(1) and setsid(1) where the machine has them (the box); on the Mac the flock stand-in
  # of tools/ref/card-tests/bin and a setsid that only execs (a runner is never taken on the Mac, so
  # the session it starts is not what the cases read). The lease is a temp file, never the machine's.
  mkdir "$t/tbin"
  command -v flock > /dev/null 2>&1 || ln -s "$LEASE_TREE/tools/ref/card-tests/bin/flock" "$t/tbin/flock"
  command -v setsid > /dev/null 2>&1 || printf '#!/bin/sh\nexec "$@"\n' > "$t/tbin/setsid"
  chmod +x "$t/tbin/"* 2> /dev/null || true
  PATH="$t/tbin:$PATH"
  if ! command -v flock > /dev/null 2>&1; then
    echo "skip trace: no flock(1) and no tools/ref/card-tests/bin/flock stand-in"
  else
    unset BLOOMERY_LEASE_TRACE BLOOMERY_Q_JOB TIMING_GPU TIMING_GPU2
    LEASE_LOCK=$t/tlock
    # trace_free_wait <trace> <id>: the `free` line of <id> within 2 s, else the empty string.
    trace_free_wait() {
      local i
      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        grep " free id=$2 " "$1" 2> /dev/null && return 0
        sleep 0.1
      done
      return 0
    }
    # A take in a subshell that holds descriptor 9 as a runner does, and ends: its output in OUT.
    trace_take() {
      OUT=$(exec 9>"$LEASE_LOCK" && flock -n 9 && lease_trace_take 2>&1; echo "id=$LEASE_TRACE_ID")
    }
    BLOOMERY_LEASE_TRACE=$t/trace1.log BLOOMERY_LEASE_CARD=docs/cards/x.card
    export BLOOMERY_LEASE_TRACE BLOOMERY_LEASE_CARD
    trace_take
    id=${OUT##*id=}
    tlines=$(grep -vc '^#' "$t/trace1.log")
    check trace-take-one-line "$tlines" 1
    matches trace-header '^# The machine lease.s trace' "$(head -n 1 "$t/trace1.log")"
    take=$(grep ' take id=' "$t/trace1.log")
    matches trace-take-fields "^[0-9T:Z-]+ take id=$id epoch=[0-9]+ pid=$$ owner=${LEASE_TREE##*/} runner=lease.sh timing_gpu=none card_file=docs/cards/x.card\$" "$take"
    check trace-take-size "$([ "${#take}" -le 400 ] && echo le400 || echo "${#take}")" le400
    # The waiter: the runner (the subshell above) is gone, the lease file turns free, the pair closes.
    free=$(trace_free_wait "$t/trace1.log" "$id")
    matches trace-free-by-waiter "^[0-9T:Z-]+ free id=$id epoch=[0-9]+ held_s=[0-9]+ by=waiter\$" "$free"
    # The waiter holds no descriptor 9: the lease is free right after the runner exits, not when the
    # waiter does (it lives until it has probed and written; a held lease here is its descriptor).
    (exec 9>"$t/tlock2" && flock -n 9 && LEASE_LOCK=$t/tlock2 lease_trace_take)
    RC=0
    lease_free "$t/tlock2" 2> /dev/null || RC=$?
    check trace-waiter-frees-lease "$RC" 0
    # A take whose runner took a queue job and two cards names them.
    BLOOMERY_LEASE_TRACE=$t/trace2.log BLOOMERY_Q_JOB=q42 TIMING_GPU=GPU-a TIMING_GPU2=GPU-b
    export BLOOMERY_LEASE_TRACE BLOOMERY_Q_JOB TIMING_GPU TIMING_GPU2
    trace_take
    id=${OUT##*id=}
    matches trace-take-job-cards " take id=$id .* owner=${LEASE_TREE##*/} job=q42 runner=lease.sh timing_gpu=GPU-a,GPU-b card_file=" "$(cat "$t/trace2.log")"
    trace_free_wait "$t/trace2.log" "$id" > /dev/null
    unset BLOOMERY_Q_JOB TIMING_GPU TIMING_GPU2
    # An unpaired take (its waiter died) is closed by the next take, before that take's own line.
    BLOOMERY_LEASE_TRACE=$t/trace3.log
    export BLOOMERY_LEASE_TRACE
    printf '# header\n2000-01-01T00:00:00Z take id=100.7 epoch=100 pid=7 owner=x runner=r timing_gpu=none card_file=-\n' > "$t/trace3.log"
    trace_take
    id=${OUT##*id=}
    trace_free_wait "$t/trace3.log" "$id" > /dev/null
    order=$(grep -v '^#' "$t/trace3.log" | awk '{print $2 " " $3}' | tr '\n' ',')
    check trace-next-take-order "$order" "take id=100.7,free id=100.7,take id=$id,free id=$id,"
    matches trace-next-take-line '^[0-9T:Z-]+ free id=100\.7 epoch=[0-9]+ held_s=[0-9]+ by=next-take$' "$(cat "$t/trace3.log")"
    # A waiter that finds its `free` already written (the next take closed it) adds none.
    printf '2000-01-01T00:00:00Z free id=100.7 epoch=200 held_s=100 by=next-take\n' >> "$t/trace3.log"
    __lease_trace_waiter "$t/tlock4" "$t/trace3.log" 100.7 100
    check trace-waiter-no-duplicate "$(grep -c ' free id=100.7 ' "$t/trace3.log")" 2
    # A write that fails: one named line, the take goes on (rc 0, no id, no waiter).
    BLOOMERY_LEASE_TRACE=$t/no-such-dir/trace.log
    export BLOOMERY_LEASE_TRACE
    RC=0
    (exec 9>"$t/tlock3" && flock -n 9 && lease_trace_take 2> "$t/err" && echo "id=[$LEASE_TRACE_ID]" > "$t/out") || RC=$?
    check trace-write-fail-rc "$RC" 0
    matches trace-write-fail-line '^\[lease\] trace not written: .*No such file or directory \(.*/no-such-dir/trace.log\); the lease goes on$' "$(cat "$t/err")"
    check trace-write-fail-no-id "$(cat "$t/out")" 'id=[]'
    # No trace file and a lock that is not the machine's: nothing written, nothing named.
    unset BLOOMERY_LEASE_TRACE
    check trace-file-other-lock "$(__lease_trace_file)" ''
    check trace-file-machine-lock "$(LEASE_LOCK=/root/bloomery-cpu.lock __lease_trace_file)" /root/bloomery-sittings.log
    # let_go: the line of the id this process took; none when nothing was written.
    LEASE_TRACE_ID=9.9 LEASE_TRACE_FILE=$t/trace4.log
    lease_trace_let_go 1
    matches trace-let-go '^[0-9T:Z-]+ let_go id=9\.9 still_held=1$' "$(cat "$t/trace4.log")"
    LEASE_TRACE_ID='' LEASE_TRACE_FILE=$t/trace5.log
    lease_trace_let_go 0
    check trace-let-go-none "$([ -e "$t/trace5.log" ] && echo written || echo none)" none
    # lease_take and lease-hold.sh write them. The machine's own lease refuses a lock override
    # (lease_take, exit 64), so these two take a lock of their own only where there is no machine lease.
    if [ -e /root/bloomery-cpu.lock ]; then
      echo "skip trace wiring: /root/bloomery-cpu.lock exists, a lease_take under another lock is refused; these two cases run off the box"
    else
      wenv=(env -u BLOOMERY_LEASE_HELD -u ROUNDS -u BLOOMERY_AB_ROUNDS BLOOMERY_LEASE_CARD=docs/cards/selftest-lease-hold.card
        BLOOMERY_LEASE_LOCK="$t/wlock" BLOOMERY_LEASE_TRACE="$t/wtrace.log")
      OUT=$("${wenv[@]}" bash -c 'source "$1" && lease_take 2>&1' _ "$LEASE_TREE/tools/ref/lease.sh")
      id=$(grep -v '^#' "$t/wtrace.log" | grep ' take id=' | sed 's/.* take id=\([^ ]*\) .*/\1/')
      matches wiring-take-card "^[0-9T:Z-]+ take id=[0-9]+\.[0-9]+ .* card_file=docs/cards/selftest-lease-hold\.card\$" "$(cat "$t/wtrace.log")"
      matches wiring-free-by-waiter " free id=$id .* by=waiter\$" "$(trace_free_wait "$t/wtrace.log" "$id")"
      # The runner's name is built, not written: tools/gate-batch.sh's script walk follows a script a
      # file names, and tools/check-recipes.sh, which runs this self-test, must not read as a taker.
      hold=lease-hold
      OUT=$("${wenv[@]}" BLOOMERY_LEASE_LOCK="$t/hlock" BLOOMERY_LEASE_TRACE="$t/htrace.log" "$LEASE_TREE/tools/ref/$hold.sh" --card docs/cards/selftest-lease-hold.card -- true 2>&1)
      id=$(grep -v '^#' "$t/htrace.log" | grep ' take id=' | sed 's/.* take id=\([^ ]*\) .*/\1/')
      trace_free_wait "$t/htrace.log" "$id" > /dev/null
      # let_go and the waiter's free race by a few ms: the set of kinds, not their order.
      check wiring-hold-kinds "$(grep -v '^#' "$t/htrace.log" | awk '{print $2}' | sort | tr '\n' ,)" free,let_go,take,
      matches wiring-hold-let-go " let_go id=$id still_held=0\$" "$(cat "$t/htrace.log")"
    fi
    unset BLOOMERY_LEASE_TRACE BLOOMERY_LEASE_CARD
  fi
  # ---- end the trace ----
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
