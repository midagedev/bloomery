#!/usr/bin/env bash
# The per-crate unsafe ratchet (Mac, grep only, no build): each crate's counts against its pins in
# tools/unsafe-ratchet.txt. This file owns the counting rule:
#   blocks  `unsafe {` lines (with or without the space) and `unsafe extern` block lines, with a
#           quoted ABI (`unsafe extern "C" {`) or the edition-2024 quoteless form (`unsafe extern {`)
#   fns     `unsafe fn` lines (pub, pub(crate), const and `unsafe extern "C" fn` forms included)
#   impls   `unsafe impl` lines
# over every .rs file under crates/<dir>/{src,tests,benches} plus the crate's own crates/<dir>/build.rs.
# The crate set is written as an exclusion, not a list, so a
# new crate is never silently left out: every directory under crates/ except oxide-ice-unroll (the reproducer outside
# the workspace). The search is by line, so comments and doc comments that name the words count too; a ratchet only
# needs to be deterministic.
# Exit: 0 every count at or under its pin (a "can come down" line for each one under). 1 a count above its pin, naming
# the crate and both numbers. 70 no ratchet file, a line that is not `<crate> <blocks> <fns> <impls>`, a crate
# missing from it, or a crate it names that this check does not count (no silent pass).
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile; the pins sit in parallel indexed arrays.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LC_ALL=C
RATCHET=tools/unsafe-ratchet.txt
BLOCK='unsafe[[:space:]]*\{|unsafe[[:space:]]+extern([[:space:]]+"[^"]*")?[[:space:]]*\{'
FN='unsafe[[:space:]]+(extern[[:space:]]+"[^"]*"[[:space:]]+)?fn'
IMPL='unsafe[[:space:]]+impl'

say() { printf 'check-unsafe: %s\n' "$*"; }
die70() { say "$*" >&2; exit 70; }

[ -f "$RATCHET" ] || die70 "no ratchet file $RATCHET to hold the per-crate unsafe counts to"

# The crates: the directories under crates/ less the reproducer (the root Cargo.toml's members).
crates=()
while IFS= read -r d; do
  case ${d#crates/} in oxide-ice-unroll) ;; *) crates+=("${d#crates/}") ;; esac
done < <(find crates -mindepth 1 -maxdepth 1 -type d | sort)
[ ${#crates[@]} -gt 0 ] || die70 "no crate directory under crates/ — nothing to count"

# count PAT FILES… — the number of lines matching PAT (0 with no file).
count() {
  local pat=$1
  shift
  [ $# -gt 0 ] || { echo 0; return 0; }
  grep -hE -- "$pat" "$@" | wc -l | tr -d '[:space:]' || true
}

# collect CRATE — every .rs under the crate's three directories plus its own build.rs, sorted, into
# rs_files (global).
rs_files=()
collect() {
  rs_files=()
  local base f
  for base in "crates/$1/src" "crates/$1/tests" "crates/$1/benches"; do
    [ -d "$base" ] || continue
    while IFS= read -r f; do rs_files+=("$f"); done < <(find "$base" -type f -name '*.rs' | sort)
  done
  # The build script sits at the crate root, outside src/, and is as much the
  # crate's code as anything under it.
  if [ -f "crates/$1/build.rs" ]; then
    rs_files+=("crates/$1/build.rs")
  fi
}

# has NAME LIST… — 0 when NAME is in LIST.
has() {
  local n=$1 a
  shift
  for a in ${1+"$@"}; do [ "$a" = "$n" ] && return 0; done
  return 1
}

# The ratchet file: '#' lines and blank lines are its rule; every other line is one crate.
pin_names=() pin_b=() pin_f=() pin_i=()
bad=''
while IFS= read -r row; do
  [ -n "$row" ] || continue
  ln=${row%%:*}
  txt=${row#*:}
  set -- $txt
  if [ $# -ne 4 ] || [ -z "$1" ]; then
    bad="$bad
$RATCHET:$ln: not '<crate> <blocks> <fns> <impls>': $txt"
    continue
  fi
  case $1 in *[!A-Za-z0-9_-]*)
    bad="$bad
$RATCHET:$ln: '$1' is not a crate directory name"
    continue
    ;;
  esac
  for v in "$2" "$3" "$4"; do
    case $v in '' | *[!0-9]*)
      bad="$bad
$RATCHET:$ln: '$v' is not a count"
      continue 2
      ;;
    esac
  done
  if has "$1" ${pin_names[@]+"${pin_names[@]}"}; then
    bad="$bad
$RATCHET:$ln: crate '$1' appears twice"
    continue
  fi
  pin_names+=("$1") pin_b+=("$2") pin_f+=("$3") pin_i+=("$4")
done <<<"$(grep -nvE '^[[:space:]]*(#|$)' "$RATCHET" || true)"
[ -z "$bad" ] || die70 "malformed ratchet:$bad"
[ ${#pin_names[@]} -gt 0 ] || die70 "$RATCHET holds no crate line — one '<crate> <blocks> <fns> <impls>' per workspace crate"

for n in ${pin_names[@]+"${pin_names[@]}"}; do
  has "$n" ${crates[@]+"${crates[@]}"} ||
    die70 "$RATCHET names crate '$n', which this check does not count — the counted set is every directory under crates/ except oxide-ice-unroll"
done
for c in ${crates[@]+"${crates[@]}"}; do
  has "$c" ${pin_names[@]+"${pin_names[@]}"} ||
    die70 "crate '$c' has no line in $RATCHET — seed its counts"
done

# pin_index CRATE — its index in pin_names (the two checks above make it exist).
pin_index() {
  local i
  for i in ${!pin_names[@]}; do
    [ "${pin_names[$i]}" = "$1" ] && { echo "$i"; return 0; }
  done
  die70 "internal: no pin row for $1"
}

printf '%-14s %6s %5s %5s  %-15s %s\n' crate blocks fns impls pin verdict
tot_b=0 tot_f=0 tot_i=0 low=0 high=0
for c in ${crates[@]+"${crates[@]}"}; do
  collect "$c"
  b=$(count "$BLOCK" ${rs_files[@]+"${rs_files[@]}"})
  f=$(count "$FN" ${rs_files[@]+"${rs_files[@]}"})
  m=$(count "$IMPL" ${rs_files[@]+"${rs_files[@]}"})
  tot_b=$((tot_b + b)) tot_f=$((tot_f + f)) tot_i=$((tot_i + m))
  pi=$(pin_index "$c")
  pb=${pin_b[$pi]} pf=${pin_f[$pi]} pm=${pin_i[$pi]}
  state='= pin'
  an=(blocks fns impls)
  av=("$b" "$f" "$m")
  ap=("$pb" "$pf" "$pm")
  for k in 0 1 2; do
    if [ "${av[$k]}" -gt "${ap[$k]}" ]; then
      say "$c ${an[$k]} ${av[$k]} above the pin ${ap[$k]} in $RATCHET — red" >&2
      state=ABOVE
      high=$((high + 1))
    elif [ "${av[$k]}" -lt "${ap[$k]}" ]; then
      say "$c ${an[$k]} ${av[$k]} below the pin ${ap[$k]} in $RATCHET — the recorded value can come down"
      state='can come down'
      low=$((low + 1))
    fi
  done
  printf '%-14s %6d %5d %5d  %-15s %s\n' "$c" "$b" "$f" "$m" "$pb/$pf/$pm" "$state"
done
if [ "$high" -gt 0 ]; then
  say "$high count(s) above their pins in $RATCHET — red" >&2
  exit 1
fi
note=''
if [ "$low" -gt 0 ]; then note=", $low below their pins can come down"; fi
say "ok — ${#crates[@]} crates, $tot_b blocks, $tot_f fns, $tot_i impls$note"
