# shellcheck shell=bash
# One load for the arms that share it: the grouping and the process driver the depth runners share
# (depth-ds41.sh, depth-qwen3moe.sh). Sourced after the runner has parsed its arms.
#
# The load key. An arm of our engine runs in a process that loads the model once and then runs a list
# of arms (generate_ds41 / generate_qwen3moe --arm ... --arm-sync), the engine cleared between them
# (app::Session::clear: a prompt after the clear writes the bits it writes in a fresh process). Arms
# share a process only when everything the load reads is theirs alike: the binary, its environment
# (every lever the binaries act on is parsed once at main and consumed by the load — the placement's
# card budget, BLOOMERY_R8, CED, the prompt call's group and buffers, the pin, the step
# stats' card events — so any NAME=VALUE an arm sets is load-time), and the runner's load-time
# arguments (--place; Qwen3's --ctx, the cache height and the flash grid). The runner writes that
# string per arm into LG_KEY[i]; an empty key is an arm that runs by itself outside this driver (a
# reference engine, a bin: base binary that knows no --arm). The per-arm parts are the feed (the
# prompt ids and their count) and -n.
#
# BLOOMERY_AB_LOAD: key (unset) groups a round's arms by load key; arm runs every arm in a process of
# its own (the one-process-per-arm path, for an A/A of the clear and for debugging). An arm whose own
# NAME=VALUE list holds BLOOMERY_AB_LOAD=arm runs alone either way; the runner strips that pair before
# the binary sees the environment and keeps it in the arm's label.
#
# Order. The units of a round are the load groups and the ungrouped arms, each unit at the place of
# its first arm in the order given; round r rotates the units by r - 1 slots and the arms inside each
# unit by r - 1 slots, so every arm of a group is the first after its load in some round (the position
# bias the rotation guards is a round's first arm; the first arm after a load is the one a cold load
# term would fall on).
#
# The driver. lg_run_unit runs one unit's process beside the runner: after the load it prints one `arm`
# line per arm and waits for a line on stdin (--arm-sync), so the runner's witness blocks and guards
# stand between two arms of one load as they stand between two processes. An arm's output is its lines
# from its `arm` line to the next arm's (or the process's end); its wall runs from the go to there, its
# majflt from the go (whole) and from its `fed` line (LG_FED_RE; timed) to there. The load's lines
# before arm 0 are echoed once, under a `[load]` line. A process that fails ends its arm in flight as a failed arm
# and the arms after it re-run in a fresh process — a fault is never cleared into the next arm; a
# process that fails before its first arm fails every arm of the unit (the load is theirs). A process
# that prints nothing for BOUND seconds is killed (its timeout's pid, the one this driver started).
#
# The runner defines, before calling lg_run_unit:
#   lg_cmd <indices...>     into LG_CMD (an array): the process's command line, `--arm-sync` included,
#                           without the timeout; into LG_ENV: its environment (an array of NAME=VALUE)
#   lg_before_load          (optional) what must hold before the process starts: a guard that would
#                           see the load's own process on the card once it runs
#   lg_pre <i> <round>      the arm's guards and its witness block before it
#   lg_post <i> <round> <rc> <output> <wall s>   the arm's witness block after it and its row (or its
#                           FAIL row); MAJ_WHOLE and MAJ_TIMED are set for it
#   lg_header_lines <header>   the load lines echoed under `[load]`, from <header> (the output's lines before
#                           arm 0) on stdout, one a line: every line a kind of the binary's schema reads and
#                           the lines of no kind that ARM_PLAIN names (records.py keep; cold-blocks.sh). It
#                           returns 1 with the reason on stdout when it cannot (the echo is display only: the
#                           driver says so on stderr and goes on).
# LG_HEADER holds the running load's lines before its arm 0, unfiltered, for a row that reads one of them. LG_PREV_OUT holds, as lg_post runs, the output
# of the arm before it in the same process (empty for a process's first arm): the lines a binary prints
# after one arm's work and before the next arm's `arm` line — a clear's report — close that output, so a
# row that reads them reads them there. LG_FED_RE, the ERE of the arm's line that
# opens its measured window (MAJ_TIMED counts from it), defaults to generate_ds41's `fed` record;
# a runner sets its own after sourcing this file (depth-qwen3moe.sh).
LG_FED_RE='^fed '
# The process's bound and its cancellation are the owner's: bounded_bg.
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/../gate-bound.sh"
LG_MODE=${BLOOMERY_AB_LOAD:-key}
case $LG_MODE in
  key | arm) ;;
  *) echo "load-groups.sh: BLOOMERY_AB_LOAD is key (the default: a round's arms of one load key in one process) or arm (every arm a process of its own), got '$LG_MODE'" >&2; exit 64 ;;
esac
LG_SOLO=BLOOMERY_AB_LOAD=arm
LG_KEY=() LG_HEADER='' LG_PREV_OUT=''

# lg_strip_solo <NAME=VALUE list>: the list without BLOOMERY_AB_LOAD=arm (comma-separated in and out).
lg_strip_solo() {
  local -a kv out=()
  local e
  IFS=, read -r -a kv <<< "$1"
  for e in "${kv[@]}"; do [ "$e" = "$LG_SOLO" ] || out+=("$e"); done
  local IFS=,
  echo "${out[*]}"
}
# lg_is_solo <NAME=VALUE list>: whether the list holds BLOOMERY_AB_LOAD=arm.
lg_is_solo() { [[ ",$1," == *",$LG_SOLO,"* ]]; }
# lg_env_key <NAME=VALUE list>: the list sorted, the order an arm gives its variables in being no part of
# the load.
lg_env_key() { tr , '\n' <<< "$1" | grep -v '^$' | sort | paste -sd, -; }

# lg_units <indices...>: the units of those arms, into LG_UNITS (one space-separated index list each), in
# the order of their first arms. An arm with an empty LG_KEY, or any arm under BLOOMERY_AB_LOAD=arm, is a
# unit of its own; a key that ends in `|solo` is too.
lg_units() {
  local i j u key
  LG_UNITS=()
  local -a ukey=()
  for i in "$@"; do
    key=${LG_KEY[$i]}
    u=
    if [ -n "$key" ] && [ "$LG_MODE" = key ] && [[ $key != *'|solo' ]]; then
      for j in "${!ukey[@]}"; do [ "${ukey[$j]}" != "$key" ] || u=$j; done
    fi
    if [ -z "$u" ]; then
      u=${#LG_UNITS[@]}
      LG_UNITS+=("") ukey+=("$key")
    fi
    LG_UNITS[u]="${LG_UNITS[$u]:+${LG_UNITS[$u]} }$i"
  done
}
# lg_rotate <k> <items...>: the items rotated left by k slots.
lg_rotate() {
  local k=$1
  shift
  local n=$# i
  local -a a=("$@")
  for ((i = 0; i < n; i++)); do printf '%s ' "${a[$(((i + k) % n))]}"; done
}
# lg_round <round>: the round's units in order, one line each, the arms inside each rotated (LG_UNITS from
# lg_units).
lg_round() {
  local r=$1 u
  local -a idx
  for u in $(lg_rotate $((r - 1)) "${!LG_UNITS[@]}"); do
    read -r -a idx <<< "${LG_UNITS[$u]}"
    lg_rotate $((r - 1)) "${idx[@]}"
    echo
  done
}
# lg_grouped <i>: whether arm <i> runs through lg_run_unit (it has a load key).
lg_grouped() { [ -n "${LG_KEY[$1]}" ]; }

lg_majflt() { awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat; }

# lg_run_unit <round> <indices...>: the unit's arms in one process after one load, each between its
# lg_pre and lg_post; a failed process ends its arm in flight and the rest run in a fresh one.
lg_run_unit() {
  local r=$1
  shift
  local -a todo=("$@")
  while [ ${#todo[@]} -gt 0 ]; do
    lg_process "$r" "${todo[@]}"
    todo=("${LG_LEFT[@]}")
  done
}

# lg_process <round> <indices...>: one process over those arms; into LG_LEFT the arms it did not reach
# after a failure (empty when it ran them all or failed before its first arm). The process reads its
# go lines from one FIFO and writes its lines into another; its pid is the timeout's (bounded_bg,
# tools/gate-bound.sh, which owns its cancellation), started here.
lg_process() {
  local r=$1
  shift
  local -a arms=("$@")
  local n=${#arms[@]} k=-1 line rc=0 st out='' prev_out='' header='' shown t0=0 f0=0 fed='' bound pid dir rfd wfd i
  LG_LEFT=()
  lg_cmd "${arms[@]}"
  bound=$((BOUND * (n + 1)))
  if declare -F lg_before_load > /dev/null; then lg_before_load; fi
  dir=$(mktemp -d "${TMPDIR:-/tmp}/load-groups.XXXXXX") || exit 2
  mkfifo "$dir/go" "$dir/out" || exit 2
  echo "[load] r$r ${n} arm(s): $(for i in "${arms[@]}"; do printf '%s ' "${ARMS[$i]}"; done)- one process, one load"
  BG_IN=$dir/go BG_OUT=$dir/out bounded_bg "$bound" env "${LG_ENV[@]}" "${LG_CMD[@]}" || exit 2
  pid=$BG_PID
  # Read-write, so this open does not wait for the process's; the process's stdin sees EOF only when
  # this end closes. A pipe that cannot be opened ends the process with the run: nothing reads it.
  exec {wfd}<> "$dir/go" || { bounded_bg_stop || :; exit 2; }
  exec {rfd}< "$dir/out" || { bounded_bg_stop || :; exit 2; }
  # lg_end_arm <arm> <rc>: that arm's output and counts, to lg_post.
  lg_end_arm() {
    local f1 t1
    f1=$(lg_majflt)
    t1=$(date +%s)
    MAJ_WHOLE=$((f1 - f0)) MAJ_TIMED=
    [ -z "$fed" ] || MAJ_TIMED=$((f1 - fed))
    LG_PREV_OUT=$prev_out
    lg_post "$1" "$r" "$2" "$out" "$((t1 - t0))"
    prev_out=$out
  }
  while :; do
    st=0
    IFS= read -r -t "$BOUND" -u "$rfd" line || st=$?
    if [ "$st" != 0 ]; then
      if [ "$st" -gt 128 ]; then
        echo "[load] r$r: no line for ${BOUND} s; killing the process (pid $pid)" >&2
        bounded_bg_stop || rc=$?
      fi
      break
    fi
    if [[ $line == 'arm '* ]]; then
      if [ "$k" -ge 0 ]; then
        lg_end_arm "${arms[$k]}" 0
      else
        LG_HEADER=$header
        if [ -n "$header" ]; then
          if shown=$(lg_header_lines "$header"); then
            [ -z "$shown" ] || sed 's/^/    /' <<< "$shown"
          else
            echo "[load] r$r: the load's lines were not echoed: $shown" >&2
          fi
        fi
      fi
      k=$((k + 1))
      if [ "$k" -ge "$n" ]; then
        out="the process printed arm line $k of a list of $n: $line"$'\n'
        bounded_bg_stop || rc=$?
        break
      fi
      lg_pre "${arms[$k]}" "$r"
      out="$line"$'\n' fed=''
      t0=$(date +%s)
      f0=$(lg_majflt)
      echo go >&"$wfd"
      continue
    fi
    if [ "$k" -lt 0 ]; then
      header+="$line"$'\n'
    else
      out+="$line"$'\n'
      if [ -z "$fed" ] && [[ $line =~ $LG_FED_RE ]]; then fed=$(lg_majflt); fi
    fi
  done
  bounded_bg_wait || rc=$?
  exec {wfd}>&- {rfd}<&-
  rm -rf "$dir"
  if [ "$k" -lt 0 ]; then
    # No arm began: the load failed, and every arm of the unit fails with its output.
    [ "$rc" != 0 ] || rc=70
    for i in "${arms[@]}"; do
      lg_pre "$i" "$r"
      t0=$(date +%s) f0=$(lg_majflt) fed='' out="$header" prev_out=''
      lg_end_arm "$i" "$rc"
    done
    return 0
  fi
  if [ "$k" -ge "$n" ]; then
    k=$((n - 1))
    [ "$rc" != 0 ] || rc=70
  elif [ "$rc" = 0 ] && [ $((k + 1)) != "$n" ]; then
    out+="the process exited 0 after arm $k of $n"$'\n'
    rc=70
  fi
  lg_end_arm "${arms[$k]}" "$rc"
  [ "$rc" = 0 ] && return 0
  LG_LEFT=("${arms[@]:$((k + 1))}")
  [ ${#LG_LEFT[@]} -eq 0 ] || echo "[load] r$r: arm ${ARMS[${arms[$k]}]} failed (rc $rc); the ${#LG_LEFT[@]} arm(s) after it run in a fresh load"
}

# `bash tools/ref/load-groups.sh --self-test`: the grouping and the order on fixed keys, and the idle bound on
# a stalled arm (just check-recipes runs it on the Mac; the driver's runs are depth-ds41-stub.sh's, on the box).
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0
  check() {
    if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi
  }
  ARMS=(a0 a1 a2 a3 a4 a5)
  LG_KEY=("bin|A" "" "bin|A" "bin|B" "bin|A|solo" "bin|B")
  LG_MODE=key
  lg_units 0 1 2 3 4 5
  check units "$(printf '%s;' "${LG_UNITS[@]}")" "0 2;1;3 5;4;"
  check round1 "$(lg_round 1 | tr '\n' ';')" "0 2 ;1 ;3 5 ;4 ;"
  check round2 "$(lg_round 2 | tr '\n' ';')" "1 ;5 3 ;4 ;2 0 ;"
  check round3 "$(lg_round 3 | tr '\n' ';')" "3 5 ;4 ;0 2 ;1 ;"
  LG_MODE=arm
  lg_units 0 1 2 3 4 5
  check units-arm "$(printf '%s;' "${LG_UNITS[@]}")" "0;1;2;3;4;5;"
  LG_MODE=key
  check strip "$(lg_strip_solo "X=1,BLOOMERY_AB_LOAD=arm,Y=2")" "X=1,Y=2"
  check strip-none "$(lg_strip_solo "X=1")" "X=1"
  lg_is_solo "X=1,BLOOMERY_AB_LOAD=arm" && s=1 || s=0
  check solo "$s" 1
  lg_is_solo "X=1" && s=1 || s=0
  check not-solo "$s" 0
  check env-key "$(lg_env_key "Y=2,X=1")" "X=1,Y=2"
  # lg_process on a stalled arm: the process prints its arm line and then nothing, so the idle bound of BOUND s (the
  # idle second, scaled with the load: tools/bloomery/selftest.py) sends its timeout a TERM, which ends it with 143,
  # and the arms behind it are left for a fresh load. The outer bound, BOUND x (arms + 1), would end it with 124.
  # A host with no timeout takes tools/ref/stub-timeout.py, which ends the same two ways.
  stall_case() {
    local sd child_status want s errexit st
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    sd=$(mktemp -d "${TMPDIR:-/tmp}/load-groups-selftest.XXXXXX")
    command -v timeout > /dev/null || cp "${BASH_SOURCE[0]%/*}/stub-timeout.py" "$sd/timeout"
    lg_majflt() { echo 0; }
    lg_pre() { :; }
    lg_post() { echo "post $1 rc=$3" >> "$sd/post.log"; }
    lg_cmd() { LG_CMD=(sh -c 'echo "arm s0"; exec sleep 1000'); LG_ENV=(); }
    ARMS=(s0 s1 s2)
    BOUND=$(python3 "${BASH_SOURCE[0]%/*}/../bloomery/selftest.py" 1) || exit 2
    (PATH="$sd:$PATH"; lg_process 1 0 1 2; echo "left=${LG_LEFT[*]}" >> "$sd/post.log") > /dev/null 2> "$sd/err.log"
    check stall-killed "$(grep -c "no line for $BOUND s; killing the process" "$sd/err.log")" 1
    check stall-term "$(grep '^post 0 ' "$sd/post.log")" "post 0 rc=143"
    check stall-left "$(grep '^left=' "$sd/post.log")" "left=1 2"
    for child_status in 0 7; do
      want=70
      [ "$child_status" = 0 ] || want=$child_status
      lg_cmd() { LG_CMD=(sh -c 'echo "arm s0"; read -r go; read -r go; exit "$1"' _ "$child_status"); LG_ENV=(); }
      : > "$sd/post.log"
      (
        lease_stop() { echo stop >&"$wfd"; return 70; }
        PATH="$sd:$PATH"
        lg_process 1 0
      ) > /dev/null 2> "$sd/err.log"
      check "stall-cleanup-$child_status-idle" "$(grep -c "no line for $BOUND s; killing the process" "$sd/err.log")" 1
      check "stall-cleanup-$child_status-rc" "$(grep '^post 0 ' "$sd/post.log")" "post 0 rc=$want"
    done
    for child_status in 0 7; do
      want=71
      [ "$child_status" = 0 ] || want=$child_status
      lg_cmd() { LG_CMD=(sh -c 'echo "arm s0"; read -r go; echo "arm extra"; read -r go; exit "$1"' _ "$child_status"); LG_ENV=(); }
      : > "$sd/post.log"
      (
        lease_stop() { echo stop >&"$wfd"; return 71; }
        PATH="$sd:$PATH"
        lg_process 1 0
      ) > /dev/null 2> "$sd/err.log"
      check "extra-arm-cleanup-$child_status-rc" "$(tail -n 1 "$sd/post.log")" "post 0 rc=$want"
    done
    for s in TERM HUP; do
      case $s in TERM) want=143 ;; HUP) want=129 ;; esac
      for errexit in off on; do
        st=0
        TMPDIR="$sd" bash -c '
          source "$1"
          [ "$3" = off ] || set -e
          reaped_signal=$2
          ARMS=(s0 s1) BOUND=5
          lg_majflt() { echo 0; }
          lg_pre() { :; }
          lg_post() { :; }
          lg_cmd() {
            [ "$#" != 1 ] || echo NEXT_ARM_STARTED
            LG_CMD=(sh -c '\''echo "arm s0"; read -r go'\''); LG_ENV=()
          }
          set -T
          trap '\''if [ "${FUNCNAME[0]:-}" = __bounded_bg_end ] &&
            [ "$BASH_COMMAND" = "eval \"\$BG_TRAPS\"" ] && [ -z "$BG_PID" ]; then
              trap - DEBUG
              kill -s "$reaped_signal" $$
            fi'\'' DEBUG
          lg_run_unit 1 0 1
        ' _ "${BASH_SOURCE[0]}" "$s" "$errexit" > "$sd/reaped.log" 2>&1 || st=$?
        check "reaped-$s-$errexit-rc" "$st" "$want"
        check "reaped-$s-$errexit-no-next-arm" "$(grep -c NEXT_ARM_STARTED "$sd/reaped.log")" 0
      done
    done
    rm -rf "$sd"
  }
  # lg_process opens its descriptors with `exec {fd}<>`, which needs bash 4.1: the macOS bash 3.2 skips the case.
  if [ "${BASH_VERSINFO[0]}" -gt 4 ] || { [ "${BASH_VERSINFO[0]}" = 4 ] && [ "${BASH_VERSINFO[1]}" -ge 1 ]; }; then
    stall_case
  else
    echo "skip stall: bash ${BASH_VERSION} has no exec {fd}<>"
  fi
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
