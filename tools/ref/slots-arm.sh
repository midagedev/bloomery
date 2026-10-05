# shellcheck shell=bash
# The aggregate arm in the depth runners that take one (depth-qwen3moe.sh, depth-ds41.sh): `<arm>@
# BLOOMERY_GEN_SLOTS=N[,NAME=VALUE...]`, whose binary decodes N streams in one pass, a row a slot, fed N·D
# ids it cuts into N windows. What is here, each runner's by its own words, with no model's branch: the
# lever refused in the runner's own environment, the arm's N, its plain twin's label, the aggregate's FAIL
# clauses over its `time pass … kind=slots` records, the residency clause over its `residency pass`
# records, and the aggregate labels kept out of the plain tables. Reading the records (records.py, by kind
# and field, through the runner's own schema) and the row itself (its columns, its window) stay the
# runner's. Sourced by the runner before it parses its arms. `bash tools/ref/slots-arm.sh --self-test` runs
# its cases on fixed columns, no box (check-recipes runs it).
#
# The runner sets, before it calls a function here:
#   SLOTS_RUNNER   the runner's name, which opens every refusal here (`<runner>: …`, `<runner>: arm '<arm>':
#                  …`, the shape of each runner's arm_refuse)
#   SLOT_LABELS    (not_slots) the aggregate arms' labels, one a line

# slots_env_check: the lever in the runner's own environment, refused by name (exit 64): every arm would
# inherit it, and a plain arm's row would read a per stream rate under a plain label.
slots_env_check() {
  [ -n "${BLOOMERY_GEN_SLOTS+x}" ] || return 0
  echo "$SLOTS_RUNNER: BLOOMERY_GEN_SLOTS=$BLOOMERY_GEN_SLOTS is set in the runner's own environment, which every arm inherits: name it per arm (<arm>@BLOOMERY_GEN_SLOTS=N), whose row reads the aggregate" >&2
  exit 64
}
# arm_slots <arm> <its NAME=VALUE list>: N into ASLOTS, empty when the list names none or N = 1 (a plain lever
# arm); an N that is not a whole number is refused by name (exit 64), the feed being N·D ids. Its range is the
# binary's: at_main and the body's pass refuse it by name, a FAIL row.
arm_slots() {
  local e
  local -a kv=()
  ASLOTS=''
  [ -z "$2" ] || IFS=, read -r -a kv <<< "$2"
  for e in ${kv[@]+"${kv[@]}"}; do
    case $e in BLOOMERY_GEN_SLOTS=*) ASLOTS=${e#*=} ;; esac
  done
  case $ASLOTS in
    '') return 0 ;;
    0* | *[!0-9]*)
      echo "$SLOTS_RUNNER: arm '$1': BLOOMERY_GEN_SLOTS=$ASLOTS is no slot count (a whole number from 1, no leading zero), and the arm feeds N·D ids" >&2
      exit 64
      ;;
  esac
  [ "$ASLOTS" != 1 ] || ASLOTS=''
}
# slots_twin <label> [<its @ list as given>]: an aggregate arm's plain twin into TWIN, the label with its list
# less the BLOOMERY_GEN_SLOTS item (`prose@BLOOMERY_GEN_SLOTS=2,X=1` -> `prose@X=1`, `ours@BLOOMERY_GEN_SLOTS=2`
# -> `ours`, `ours@prose@BLOOMERY_GEN_SLOTS=2` -> `ours@prose`, `bin:t@prose@BLOOMERY_GEN_SLOTS=2` ->
# `bin:t@prose`). With no list given, the list is the label's last `@` item: a value holds no `@`.
slots_twin() {
  local e rest='' list=${2-${1##*@}}
  local -a kv=()
  IFS=, read -r -a kv <<< "$list"
  for e in ${kv[@]+"${kv[@]}"}; do
    case $e in BLOOMERY_GEN_SLOTS=*) ;; *) rest+=${rest:+,}$e ;; esac
  done
  TWIN=${1%@"$list"}${rest:+@$rest}
}
# slots_count <N, empty for a plain arm> <ms> <positions> <kind> <warm>: the arm's `time pass` records, one
# column each (one record a line, as records.py's `*` fields give them), into SL_N, SL_POS and SL_MS (the
# counted kind=slots rounds, `warm` left out: their number, positions and ms), SL_BAD (those whose positions
# is not N) and SL_ALL and SL_ALL_MS (every kind=slots record and its ms). Returns 1 with FAIL_WHY when a
# clause fails: a plain arm with any kind=slots record, an aggregate arm with no counted one, or one whose
# positions is not N.
slots_count() {
  local n=$1
  FAIL_WHY=''
  read -r SL_N SL_POS SL_MS SL_BAD SL_ALL SL_ALL_MS < <(paste -d' ' <(printf '%s\n' "$2") <(printf '%s\n' "$3") \
    <(printf '%s\n' "$4") <(printf '%s\n' "$5") | awk -v want="${n:-0}" '$3 == "slots" {
      all++; all_ms += $1
      if ($4 == "1") next
      c++; p += $2; ms += $1; if ($2 != want) bad++
    } END { printf "%d %d %.4f %d %d %.4f\n", c, p, ms, bad, all, all_ms }')
  if [ -z "$n" ]; then
    [ "$SL_ALL" != 0 ] || return 0
    FAIL_WHY="its output holds $SL_ALL time pass record(s) of kind=slots and the arm names no BLOOMERY_GEN_SLOTS: its row would read one stream's rate off several"
    return 1
  fi
  if [ "$SL_N" = 0 ]; then
    FAIL_WHY="the arm runs BLOOMERY_GEN_SLOTS=$n and printed no counted time pass record of kind=slots: no aggregate to read"
    return 1
  fi
  if [ "$SL_BAD" != 0 ]; then
    FAIL_WHY="$SL_BAD of its $SL_N counted kind=slots records hold positions other than its $n slots"
    return 1
  fi
}
# slots_residency <N> <residency host word> <pass> <kept> <boundary>: the residency clause over an aggregate
# arm's `residency pass` records, one column each, the first the seed's none/0. A boundary's report names the
# pass it ended, so after the seed's come the prompt calls (pass=prompt kept=0, at least one a slot: a
# slot's prompt may take more than one call) up to the first pass=slots, then only pass=slots kept=N, one a
# round; a step a slot or a pair in place of the pass fails, naming the record. An arm with no residency
# host record (the residency off, or a placement that resolves it off) has no residency pass to read and
# passes the clause, its passes pinned by slots_count's `time pass … kind=slots` records alone; residency pass
# records under such an arm fail it by name. Returns 1 with FAIL_WHY when it fails.
slots_residency() {
  local n=$1 v what b p k m s
  FAIL_WHY=''
  if [ -z "$2" ]; then
    [ -z "$3" ] && return 0
    FAIL_WHY="the slots residency clause: residency pass records under an arm with no residency host record (BLOOMERY_RESIDENCY off, or unset under a placement that resolves it off)"
    return 1
  fi
  v=$(paste -d' ' <(printf '%s\n' "$3") <(printf '%s\n' "$4") <(printf '%s\n' "$5") | awk -v n="$n" '
    NR == 1 { next }
    $1 == "slots" { in_slots = 1 }
    !in_slots { if ($1 != "prompt" || $2 != 0) { print "prompt", $3, $1, $2; bad = 1; exit } m++; next }
    { s++; if ($1 != "slots" || $2 != n) { print "slots", $3, $1, $2; bad = 1; exit } }
    END { if (!bad) print "ok", m + 0, s + 0 }')
  read -r what b p k <<< "$v"
  case $what in
    prompt) FAIL_WHY="the slots residency clause: the residency pass at boundary=$b reads pass=$p kept=$k before its first pass=slots, where only its $n slots' prompt calls (pass=prompt kept=0) stand" ;;
    slots) FAIL_WHY="the slots residency clause: the residency pass at boundary=$b reads pass=$p kept=$k, not pass=slots kept=$n: a decode pass that was not one pass of its $n slots" ;;
    ok)
      m=$b s=$p
      if [ "$s" = 0 ]; then
        FAIL_WHY="the slots residency clause: no pass=slots kept=$n record after its $n slots' prompt calls: no decode pass of its $n slots reached the residency"
      elif [ "$m" -lt "$n" ]; then
        FAIL_WHY="the slots residency clause: $m pass=prompt kept=0 record(s) before its first pass=slots, fewer than one a slot of its $n"
      else
        return 0
      fi
      ;;
    *) FAIL_WHY="the slots residency clause: its residency pass records did not read ($v)" ;;
  esac
  return 1
}
# not_slots: the labels on stdin less the aggregate arms' (SLOT_LABELS), which have their own table. The
# labels reach awk as a file of at least one line (an awk -v value holds no newline in every awk).
not_slots() { awk 'NR == FNR { if ($0 != "") x[$0] = 1; next } !($0 in x)' <(printf '%s\n' "$SLOT_LABELS") -; }

# `bash tools/ref/slots-arm.sh --self-test`: each function on fixed arms, labels and record columns (bash 3.2
# on the Mac); one line per check, `ok <name>` or `FAIL <name>: got … want …`, the verdict last.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0 checks=0
  check() {
    checks=$((checks + 1))
    if [ "$2" = "$3" ]; then
      echo "ok $1"
    else
      echo "FAIL $1: got [$2] want [$3]"
      fails=$((fails + 1))
    fi
  }
  SLOTS_RUNNER=runner.sh
  # The environment's lever and the arm's N (a whole number from 1; 1 a plain lever arm).
  out=$(BLOOMERY_GEN_SLOTS=2 slots_env_check 2>&1) && r=0 || r=$?
  check env "$r|${out%%,*}" "64|runner.sh: BLOOMERY_GEN_SLOTS=2 is set in the runner's own environment"
  out=$(unset BLOOMERY_GEN_SLOTS; slots_env_check 2>&1) && r=0 || r=$?
  check env-unset "$r|$out" "0|"
  arm_slots 6@X=1,BLOOMERY_GEN_SLOTS=2 X=1,BLOOMERY_GEN_SLOTS=2
  check slots-two "$ASLOTS" 2
  arm_slots 6@BLOOMERY_GEN_SLOTS=1 BLOOMERY_GEN_SLOTS=1
  check slots-one "$ASLOTS" ''
  arm_slots 6 ''
  check slots-none "$ASLOTS" ''
  for bad in x 02 0 -1; do
    out=$(arm_slots "6@BLOOMERY_GEN_SLOTS=$bad" "BLOOMERY_GEN_SLOTS=$bad" 2>&1) && r=0 || r=$?
    check "slots-bad-$bad" "$r|${out%%,*}" "64|runner.sh: arm '6@BLOOMERY_GEN_SLOTS=$bad': BLOOMERY_GEN_SLOTS=$bad is no slot count (a whole number from 1"
  done
  # The plain twin, from the label alone or with the arm's list as given.
  for c in 'ours@BLOOMERY_GEN_SLOTS=2|ours' 'prose@BLOOMERY_GEN_SLOTS=2,X=1|prose@X=1' \
    'prose@place=gate,BLOOMERY_GEN_SLOTS=2|prose@place=gate' 'ours@prose@BLOOMERY_GEN_SLOTS=2|ours@prose' \
    'ours@prose@X=1,BLOOMERY_GEN_SLOTS=2,Y=2|ours@prose@X=1,Y=2' 'bin:t@prose@BLOOMERY_GEN_SLOTS=2|bin:t@prose' \
    'bin:t@BLOOMERY_GEN_SLOTS=2|bin:t'; do
    slots_twin "${c%%|*}"
    check "twin ${c%%|*}" "$TWIN" "${c#*|}"
  done
  slots_twin prose@place=gate,BLOOMERY_GEN_SLOTS=2 place=gate,BLOOMERY_GEN_SLOTS=2
  check twin-list "$TWIN" prose@place=gate
  # The time pass clauses: three rounds, the first warm, of two positions at 20 then 8 ms.
  ms=$'20.0000\n8.0000\n8.0000' pos=$'2\n2\n2' kind=$'slots\nslots\nslots' warm=$'1\n0\n0'
  slots_count 2 "$ms" "$pos" "$kind" "$warm" && r=0 || r=$?
  check count "$r|$SL_N $SL_POS $SL_MS $SL_BAD $SL_ALL $SL_ALL_MS" "0|2 4 16.0000 0 3 36.0000"
  slots_count 2 "$ms" $'2\n2\n1' "$kind" "$warm" && r=0 || r=$?
  check count-pos "$r|$FAIL_WHY" "1|1 of its 2 counted kind=slots records hold positions other than its 2 slots"
  slots_count 2 '' '' '' '' && r=0 || r=$?
  check count-none "$r|${FAIL_WHY%%:*}" "1|the arm runs BLOOMERY_GEN_SLOTS=2 and printed no counted time pass record of kind=slots"
  slots_count 2 '20.0000' 2 slots 1 && r=0 || r=$?
  check count-warm-only "$r|$SL_N $SL_ALL" "1|0 1"
  slots_count '' "$ms" "$pos" "$kind" "$warm" && r=0 || r=$?
  check count-leak "$r|${FAIL_WHY%%:*}" "1|its output holds 3 time pass record(s) of kind=slots and the arm names no BLOOMERY_GEN_SLOTS"
  slots_count '' '' '' '' '' && r=0 || r=$?
  check count-plain "$r|$SL_ALL|$FAIL_WHY" "0|0|"
  # The residency clause: none/0, the prompt calls, then pass=slots kept=N.
  res() {
    local p='' k='' b='' i=0 e
    for e in "$@"; do
      p+=${p:+$'\n'}${e%/*} k+=${k:+$'\n'}${e#*/} b+=${b:+$'\n'}$i
      i=$((i + 1))
    done
    slots_residency 2 "${RH-mid}" "$p" "$k" "$b" && r=0 || r=$?
    out="$r|$FAIL_WHY"
  }
  res none/0 prompt/0 prompt/0 slots/2 slots/2
  check res "$out" "0|"
  res none/0 prompt/0 prompt/0 prompt/0 slots/2
  check res-manyprompt "$out" "0|"
  res none/0 prompt/0 slots/2 slots/2
  check res-oneprompt "$out" "1|the slots residency clause: 1 pass=prompt kept=0 record(s) before its first pass=slots, fewer than one a slot of its 2"
  res none/0 prompt/0 prompt/0 step/1 slots/2
  check res-stepfirst "$out" "1|the slots residency clause: the residency pass at boundary=3 reads pass=step kept=1 before its first pass=slots, where only its 2 slots' prompt calls (pass=prompt kept=0) stand"
  res none/0 prompt/0 prompt/0 slots/2 pair/2
  check res-pair "$out" "1|the slots residency clause: the residency pass at boundary=4 reads pass=pair kept=2, not pass=slots kept=2: a decode pass that was not one pass of its 2 slots"
  res none/0 prompt/0 prompt/0 slots/2 prompt/0
  check res-promptafter "${out%%,*}" "1|the slots residency clause: the residency pass at boundary=4 reads pass=prompt kept=0"
  res none/0 prompt/0 prompt/0 slots/1
  check res-kept "${out%%,*}" "1|the slots residency clause: the residency pass at boundary=3 reads pass=slots kept=1"
  res none/0 prompt/0 prompt/0
  check res-nopass "${out%%:*}" "1|the slots residency clause"
  RH='' res
  check res-off "$out" "0|"
  RH='' res none/0 prompt/0 prompt/0 slots/2
  check res-off-records "${out%%:*}" "1|the slots residency clause"
  # The aggregate labels out of a plain table.
  SLOT_LABELS=$'ours@BLOOMERY_GEN_SLOTS=2\nprose@BLOOMERY_GEN_SLOTS=2\n'
  check not-slots "$(printf '%s\n' ours ours@BLOOMERY_GEN_SLOTS=2 prose prose@BLOOMERY_GEN_SLOTS=2 ik | not_slots | paste -sd' ' -)" "ours prose ik"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($checks checks, $fails failures)"
  [ "$fails" = 0 ]
  exit
fi
