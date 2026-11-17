#!/usr/bin/env bash
# The plan room: the one owner of the host room `just records-refresh` plans V4.1's placement (a) under.
#
# generate_ds41 --plan resolves the unset residency word against the plan's host headroom and mem_left, each the host
# room less the plan's host need. The pinned BLOOMERY_HOST_ROOM=R is the plan's room as given, on any machine: a room
# past the machine's capacity is a plan's to take (only a load refuses it), so the records at R depend on R alone, not
# on the machine's reading or its memory. They do move with R: at R = PLAN_ROOM the pick is bound by the headroom (the
# committed ds41-p128-ced-on.rec holds pool_shrunk with 33 MiB left), so a larger room is another plan. Until a room
# past PLAN_ROOM was honoured (2026-11-17) the script also planned at R + 64 GiB and refused when the two differed;
# that held only while the plan capped a room to R, and the second plan is gone with the cap.
#
#   bash tools/ref/plan-room.sh --room   R in bytes on stdout; any machine, no binary
#   bash tools/ref/plan-room.sh P C      on the box, the tree's built target/release/generate_ds41 --plan --depth P
#                                        --place a under BLOOMERY_CED=C (on or off) at BLOOMERY_HOST_ROOM=R, its stderr
#                                        passed through: its stdout. A run that fails: nothing on stdout and one line on
#                                        stderr naming P, C, the room and the exit code.
#
# R is the `pub const PLAN_ROOM` line of crates/placement/src/placement/workstation.rs, underscores dropped (the crate
# holds it to HOST_USABLE − OS_OTHER at compile time); this file holds no copy of the number.
# Exit codes: 0 R or the plan printed; 1 the run failed or there is no binary; 64 a usage error, a
# BLOOMERY_HOST_ROOM already set (the room is this script's), or a constant that is missing, given twice or not a
# whole number.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SRC=crates/placement/src/placement/workstation.rs
BIN=target/release/generate_ds41

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

ROOM=$(const PLAN_ROOM) || exit $?

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
WHAT="P=$P C=$C, room $ROOM"
if [ ! -x "$BIN" ]; then
  echo "plan-room.sh: $WHAT: no $BIN in $ROOT to run (records-refresh builds it before its plans)" >&2
  exit 1
fi

one=''
trap 'rm -f ${one:+"$one"}' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
one=$(mktemp "${TMPDIR:-/tmp}/plan-room.XXXXXX")

# The plan at BLOOMERY_HOST_ROOM=R, its stdout to $one, its stderr through. A run that fails ends the script: one line
# naming it and its exit code, exit 1.
rc=0
BLOOMERY_CED=$C BLOOMERY_HOST_ROOM=$ROOM "$BIN" --plan --depth "$P" --place a > "$one" || rc=$?
if [ "$rc" != 0 ]; then
  echo "plan-room.sh: $WHAT: $BIN --plan at BLOOMERY_HOST_ROOM=$ROOM exited $rc" >&2
  exit 1
fi
cat "$one"
echo "plan-room.sh: $WHAT: $(wc -l < "$one" | tr -d ' ') lines" >&2
