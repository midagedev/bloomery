# shellcheck shell=bash
# One load for the arms that share it: the grouping and the process driver the depth runners share
# (depth-ds41.sh, depth-qwen3moe.sh). Sourced after the runner has parsed its arms.
#
# The load key. An arm of our engine runs in a process that loads the model once and then runs a list
# of arms (generate_ds41 / generate_qwen3moe --arm ... --arm-sync), the engine cleared between them
# (app::Session::clear: a prompt after the clear writes the bits it writes in a fresh process). Arms
# share a process only when everything the load reads is theirs alike: the binary, its environment
# (every lever the binaries act on is parsed once at main and consumed by the load — the placement's
# hot list and card budget, BLOOMERY_R8, CED, the prompt call's group and buffers, the pin, the step
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
# majflt from the go (whole) and from its `fed` line (timed) to there. The load's lines before arm 0
# are echoed once, under a `[load]` line. A process that fails ends its arm in flight as a failed arm
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
# and LG_HEADER_RE, the ERE of the load lines echoed under `[load]`. LG_HEADER holds the running load's
# lines before its arm 0, for a row that reads one of them.
LG_MODE=${BLOOMERY_AB_LOAD:-key}
case $LG_MODE in
  key | arm) ;;
  *) echo "load-groups.sh: BLOOMERY_AB_LOAD is key (the default: a round's arms of one load key in one process) or arm (every arm a process of its own), got '$LG_MODE'" >&2; exit 64 ;;
esac
LG_SOLO=BLOOMERY_AB_LOAD=arm
LG_KEY=() LG_HEADER=

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
# go lines from one FIFO and writes its lines into another; its pid is the timeout's, started here.
lg_process() {
  local r=$1
  shift
  local -a arms=("$@")
  local n=${#arms[@]} k=-1 line rc=0 st out='' header='' t0=0 f0=0 fed='' bound pid dir rfd wfd i
  LG_LEFT=()
  lg_cmd "${arms[@]}"
  bound=$((BOUND * (n + 1)))
  if declare -F lg_before_load > /dev/null; then lg_before_load; fi
  dir=$(mktemp -d "${TMPDIR:-/tmp}/load-groups.XXXXXX") || exit 2
  mkfifo "$dir/go" "$dir/out" || exit 2
  echo "[load] r$r ${n} arm(s): $(for i in "${arms[@]}"; do printf '%s ' "${ARMS[$i]}"; done)- one process, one load"
  env "${LG_ENV[@]}" timeout --kill-after=10 "$bound" "${LG_CMD[@]}" < "$dir/go" > "$dir/out" 2>&1 &
  pid=$!
  # Read-write, so this open does not wait for the process's; the process's stdin sees EOF only when
  # this end closes.
  exec {wfd}<> "$dir/go"
  exec {rfd}< "$dir/out"
  # lg_end_arm <arm> <rc>: that arm's output and counts, to lg_post.
  lg_end_arm() {
    local f1 t1
    f1=$(lg_majflt)
    t1=$(date +%s)
    MAJ_WHOLE=$((f1 - f0)) MAJ_TIMED=
    [ -z "$fed" ] || MAJ_TIMED=$((f1 - fed))
    lg_post "$1" "$r" "$2" "$out" "$((t1 - t0))"
  }
  while :; do
    if ! IFS= read -r -t "$BOUND" -u "$rfd" line; then
      st=$?
      if [ "$st" -gt 128 ]; then
        echo "[load] r$r: no line for ${BOUND} s; killing the process (pid $pid)" >&2
        kill "$pid" 2> /dev/null
      fi
      break
    fi
    if [[ $line == 'arm '* ]]; then
      if [ "$k" -ge 0 ]; then
        lg_end_arm "${arms[$k]}" 0
      else
        LG_HEADER=$header
        [ -z "$header" ] || printf '%s' "$header" | grep -E "$LG_HEADER_RE" | sed 's/^/    /'
      fi
      k=$((k + 1))
      if [ "$k" -ge "$n" ]; then
        out="the process printed arm line $k of a list of $n: $line"$'\n'
        kill "$pid" 2> /dev/null
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
      if [ -z "$fed" ] && [[ $line == 'fed '* ]]; then fed=$(lg_majflt); fi
    fi
  done
  wait "$pid"
  rc=$?
  exec {wfd}>&- {rfd}<&-
  rm -rf "$dir"
  if [ "$k" -lt 0 ]; then
    # No arm began: the load failed, and every arm of the unit fails with its output.
    [ "$rc" != 0 ] || rc=70
    for i in "${arms[@]}"; do
      lg_pre "$i" "$r"
      t0=$(date +%s) f0=$(lg_majflt) fed='' out="$header"
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

# `bash tools/ref/load-groups.sh --self-test`: the grouping and the order on fixed keys (just
# check-recipes runs it on the Mac; the driver itself is depth-ds41-stub.sh's, on the box).
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
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
