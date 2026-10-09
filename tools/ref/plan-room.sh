#!/usr/bin/env bash
# The plan room: the one owner of the host room `just records-refresh` plans V4.1's placement (a) under.
#
# generate_ds41 --plan resolves the unset residency word against two bounds (crates/gpu-gates/src/residency41.rs): the
# plan's host headroom, HOST_USABLE − (experts + tables + shadows + reserves), and mem_left, the host's room less the
# plan's host need, experts + tables + shadows + reserves − OS_OTHER. At the room R = HOST_USABLE − OS_OTHER mem_left is
# the headroom exactly, so the headroom alone decides; at any larger room mem_left passes the headroom, and the pick and
# every record --plan prints stay the same. A plan whose records differ between R and R + 64 GiB moves with the room.
#
#   bash tools/ref/plan-room.sh --room   R in bytes on stdout; any machine, no binary
#   bash tools/ref/plan-room.sh P C      on the box, the tree's built target/release/generate_ds41 --plan --depth P
#                                        --place a under BLOOMERY_CED=C (on or off), once at BLOOMERY_HOST_ROOM=R and
#                                        once at R + 64 GiB, each run's stderr passed through. Both exit 0 with the same
#                                        stdout bytes: the first run's stdout. Else nothing on stdout and one line on
#                                        stderr naming P, C, both rooms and the first line that differs, or the run that
#                                        failed and its exit code.
#
# R comes from the two `pub const` lines of crates/placement/src/placement/workstation.rs, underscores dropped; this
# file holds no copy of either number.
# Exit codes: 0 R or the plan printed; 1 the two rooms' plans differ or a run failed; 64 a usage error, a
# BLOOMERY_HOST_ROOM already set (the room is this script's), or a constant that is missing, given twice or not a
# whole number.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SRC=crates/placement/src/placement/workstation.rs
BIN=target/release/generate_ds41
MORE=$((64 << 30))

usage() {
  echo "plan-room.sh: usage: bash tools/ref/plan-room.sh --room | bash tools/ref/plan-room.sh <depth> <on|off>" >&2
  exit 64
}

refuse() {
  echo "plan-room.sh: $*" >&2
  exit 64
}

# const NAME: the value of the one `pub const NAME: u64 = <digits>;` line of $SRC, underscores dropped. Missing, given
# twice (any type, any indent) or not a literal whole number of at most 18 digits: refused by name, exit 64.
const() {
  local name=$1 n line value
  [ -f "$SRC" ] || refuse "$SRC: no such file, so no $name"
  n=$(grep -cE "^[[:space:]]*pub const ${name}[[:space:]]*:" "$SRC" || true)
  case $n in
    1) ;;
    0) refuse "$SRC: no \`pub const $name\` line" ;;
    *) refuse "$SRC: \`pub const $name\` is given $n times" ;;
  esac
  line=$(grep -E "^[[:space:]]*pub const ${name}[[:space:]]*:" "$SRC")
  value=$(printf '%s\n' "$line" | sed -nE "s/^pub const ${name}: u64 = ([0-9][0-9_]*);[[:space:]]*$/\\1/p" | tr -d _)
  if [ -z "$value" ] || [ "${#value}" -gt 18 ]; then
    refuse "$SRC: \`pub const $name\` is not a u64 whole number: $line"
  fi
  echo $((10#$value))
}

usable=$(const HOST_USABLE) || exit $?
os=$(const OS_OTHER) || exit $?
[ "$usable" -gt "$os" ] || refuse "$SRC: HOST_USABLE $usable is not past OS_OTHER $os"
ROOM=$((usable - os))

if [ $# = 1 ] && [ "$1" = --room ]; then
  echo "$ROOM"
  exit 0
fi
[ $# = 2 ] || usage
P=$1 C=$2
case $P in
  '' | *[!0-9]*) refuse "depth '$P': a whole number of positions" ;;
esac
case $C in
  on | off) ;;
  *) refuse "BLOOMERY_CED '$C': on or off" ;;
esac
[ -z "${BLOOMERY_HOST_ROOM+set}" ] || refuse "BLOOMERY_HOST_ROOM is set ('$BLOOMERY_HOST_ROOM'): the plan's room is this script's"
LARGER=$((ROOM + MORE))
WHAT="P=$P C=$C, rooms $ROOM and $LARGER"
if [ ! -x "$BIN" ]; then
  echo "plan-room.sh: $WHAT: no $BIN in $ROOT to run (records-refresh builds it before its plans)" >&2
  exit 1
fi

one='' two=''
trap 'rm -f ${one:+"$one"} ${two:+"$two"}' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
one=$(mktemp "${TMPDIR:-/tmp}/plan-room.XXXXXX")
two=$(mktemp "${TMPDIR:-/tmp}/plan-room.XXXXXX")

# plan ROOM OUT: the plan at BLOOMERY_HOST_ROOM=ROOM, its stdout to OUT, its stderr through. A run that fails ends the
# script: one line naming it and its exit code, exit 1.
plan() {
  local rc=0
  BLOOMERY_CED=$C BLOOMERY_HOST_ROOM=$1 "$BIN" --plan --depth "$P" --place a > "$2" || rc=$?
  if [ "$rc" != 0 ]; then
    echo "plan-room.sh: $WHAT: $BIN --plan at BLOOMERY_HOST_ROOM=$1 exited $rc" >&2
    exit 1
  fi
}

plan "$ROOM" "$one"
plan "$LARGER" "$two"

if cmp -s "$one" "$two"; then
  cat "$one"
  echo "plan-room.sh: $WHAT: the same $(wc -l < "$one" | tr -d ' ') lines at both rooms" >&2
  exit 0
fi
first=$(awk -v a="$one" -v b="$two" -v ra="$ROOM" -v rb="$LARGER" 'BEGIN {
  for (n = 1; ; n++) {
    ga = (getline la < a) > 0
    gb = (getline lb < b) > 0
    if (!ga && !gb) exit
    if (!ga) la = "(no line)"
    if (!gb) lb = "(no line)"
    if (!ga || !gb || la != lb) {
      printf "line %d at %s: %s | at %s: %s", n, ra, la, rb, lb
      exit
    }
  }
}')
echo "plan-room.sh: $WHAT: the two rooms' plans differ: ${first:-their outputs differ past the last whole line}" >&2
exit 1
