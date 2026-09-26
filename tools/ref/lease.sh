#!/usr/bin/env bash
# shellcheck shell=bash
# The machine-wide lease and what the runners under it share. Sourced, never executed: it defines
# functions and variables and exports nothing, and nothing here may exit or fail at top level
# (most runners run under `set -e`).
#
#   source "${BASH_SOURCE[0]%/*}/lease.sh"
#   WITNESS=(head loadavg pressure-io lock-holder model)   # this runner's witness fields, in order
#   lease_take                                              # check the card, wait up to 30 min, or exit
#   witness pre; <the measured run>; witness post
#
# What it owns, so that the runners cannot drift apart:
#   the lease's take: its descriptor (9), the 30-minute wait, exit 75 when the wait runs out;
#   the read side, from tools/ref/lease-probe.sh, which this file sources first: the lease's file
#   (LEASE_LOCK), its probe (lease_free, a shared lock), the holds (/root/bloomery-<owner>-hold),
#   tools/box.sh's guard (lease_guard) and naming the lease's holder to a waiter (lease_holders);
#   the bound on every process a runner starts under the lease (lease_bounded);
#   the card every lease needs (tools/ref/card.py, at lease_take below): no card, no lease;
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
#   card            the timing card's lines: timing-card.sh's witness_card
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
# A runner that runs its arms in rounds sets ROUNDS before lease_take, and an ab card is checked at
# that count; with no ROUNDS, at BLOOMERY_AB_ROUNDS (the count of a caller that starts this runner
# once per arm, like tools/gpu-ab.py); else at the card's own `rounds`. ROUND_MINUTES, a runner's box
# minutes per round, prices a ruler refusal.
# BLOOMERY_LEASE_LOCK names another lock file, for the Mac stub tests (tools/ref/card-tests/run.sh).
# A run under any other file holds no machine lease, so where /root/bloomery-cpu.lock exists (the box)
# lease_take refuses the override (exit 64): a fake lock during a real sitting would contaminate it.
# A command that tools/ref/lease-hold.sh runs carries BLOOMERY_LEASE_HELD (that script's pid): a lease
# taken inside it would wait on the one already held and end as contention, so it exits 64 at once.
lease_take() {
  if [ -n "${BLOOMERY_LEASE_HELD:-}" ]; then
    echo "[lease] refused: this run is inside tools/ref/lease-hold.sh (pid $BLOOMERY_LEASE_HELD), which holds the lease; a second lease would wait on it" >&2
    exit 64
  fi
  if [ "$LEASE_LOCK" != /root/bloomery-cpu.lock ] && [ -e /root/bloomery-cpu.lock ]; then
    echo "[lease] refused: BLOOMERY_LEASE_LOCK=$LEASE_LOCK on the machine whose lease is /root/bloomery-cpu.lock: a run under another lock would share the box with a real sitting" >&2
    exit 64
  fi
  lease_card || exit $?
  [ "$LEASE_LOCK" = /root/bloomery-cpu.lock ] ||
    echo "[lease] BLOOMERY_LEASE_LOCK=$LEASE_LOCK is not the machine lease: nothing measured under it is admissible"
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
# runs on, and the `busiest` witness field only shows it. guard_cpu sums `ps` %CPU over the
# processes whose name is in CPU_BUSY_COMMS (BLOOMERY_CPU_BUSY_COMMS); above CPU_BUSY_PCT
# (BLOOMERY_CPU_BUSY_PCT, percent of one cpu — a chosen threshold, not a measured one) it prints
# `[cpu-busy]` on stderr and sets CPU_BUSY_TAG to ` [cpu-busy]` for the runner's row; with
# BLOOMERY_OTHER_STRICT=1 it prints a witness block and exits 75 instead, as guard_other does for a
# busy other card. `ps` %CPU is cpu time over elapsed time: a build started seconds ago reads its
# real load, a long-lived process that only now turned busy reads low. Call it only between the
# runner's own arms, when no matching process is the runner's own; the runner clears CPU_BUSY_TAG
# when a row starts.
CPU_BUSY_COMMS=${BLOOMERY_CPU_BUSY_COMMS:-cargo rustc cc1plus nvcc cicc ptxas llama-bench generate_ds41}
CPU_BUSY_PCT=${BLOOMERY_CPU_BUSY_PCT:-50}
CPU_BUSY_TAG=

# cpu_busy_reading: `<sum of %CPU> <name %CPU;...>` over the processes CPU_BUSY_COMMS names.
cpu_busy_reading() {
  ps -eo pcpu=,comm= | awk -v list=" $CPU_BUSY_COMMS " '
    index(list, " " $2 " ") { s += $1; m = m sprintf("%s %s%%;", $2, $1) }
    END { printf "%.1f %s\n", s, m }'
}

guard_cpu() {
  local tag=$1 reading sum
  case $CPU_BUSY_PCT in
    '' | *[!0-9.]* | *.*.*)
      echo "guard_cpu: BLOOMERY_CPU_BUSY_PCT is a percentage, got '$CPU_BUSY_PCT'" >&2
      exit 64
      ;;
  esac
  reading=$(cpu_busy_reading)
  sum=${reading%% *}
  awk -v s="$sum" -v t="$CPU_BUSY_PCT" 'BEGIN { exit !(s > t) }' || return 0
  echo "[cpu-busy] $(now) $tag: ${sum}% > ${CPU_BUSY_PCT}% of one cpu over [${reading#* }]" >&2
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
__witness_card() { witness_card; }
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
