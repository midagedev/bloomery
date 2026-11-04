#!/usr/bin/env bash
# bloomery — a round's compile lane: the build shapes a change touches, type-checked on the box with no card and no lease, so
# a round that may not build on its Mac still hands back code that compiles.
#
#   tools/round-check.sh [--tree DIR] [--base SPEC] [--clippy] [--fresh] [--dry-run]
#   tools/round-check.sh --self-test     the shape list, the refusals, the box command, the rc and the ledger against a stub
#                                        recipes.py and a stub box.sh; no box, no cargo (check-recipes runs it)
#
# The shapes are `tools/recipes.py combos --base SPEC` (default origin/main): every `cargo check -p <package> [--profile test]
# [--features …] <targets>` a recipe compiles whose inputs a file changed since the merge-base of SPEC and HEAD (the working
# tree and its untracked files included, so an uncommitted round counts) — the touched crates, their dependents' targets, and
# the features recipes.py resolves for each. recipes.py is the one owner of that list; this script only runs it. `--clippy`
# runs each shape as `cargo clippy` (rustc plus the workspace lints, so a deny lint is red here too). There is no
# `cargo oxide check`: a check runs no codegen backend, and `just check` is the same plain `cargo check` on the box. So a
# device crate (crates/gpu and every crate linking it) is type-checked on its host side only: its kernels' codegen, the
# PTX and the device-side errors cuda-oxide raises are not covered; they show only in a `cargo oxide` build (gate-*).
#
# The run is one tools/box.sh call of DIR (default: this tree), so everything box.sh owns holds: the sync of the tree to its
# remote directory, the hold on this side, the box-side guard (it waits while the timing lease or a hold is up and ends 75
# at its bound, BLOOMERY_BOX_WAIT), the profile and CARGO_BUILD_JOBS of a track directory. The remote directory is
# BLOOMERY_REMOTE when set, else ~/repo/bloomery-<track> where <track> is DIR's basename without a leading `bloomery-`: a
# round never builds in the main tree's ~/repo/bloomery, whose cargo build lock and target/ belong to the lead (a DIR named
# `bloomery` is refused, 64, unless BLOOMERY_REMOTE names a directory). On the box, `round-check.sh --on-box CMD…` runs every
# shape, a red one too, one `round-check: [i/N]` line before and after each, and ends with the first red shape's rc; then
# tools/lock-back.sh brings a Cargo.lock the check moved back to DIR, as `just check` does.
#
# Ledger. A shape green on the box is appended to ~/.cache/bloomery/round-check-<check|clippy>.tsv as
# `<input key><TAB>green<TAB><date><TAB><DIR>` (the key is combos' own: every file the shape reads, its command and the
# environment); the next run skips a shape green at its key, so a round re-checks only what moved and what was red. `--fresh`
# reads no ledger (every changed shape runs) and still records. BLOOMERY_ROUND_LEDGER_DIR moves the ledgers. Each run's wall
# lands in ~/.cache/bloomery/round-check-times.tsv (`<date> <DIR> <mode> <shapes> <wall s> <rc>`), and its median per shape
# predicts the next run's wall before it starts.
#
# Exit: the first red shape's rc, or box.sh's own (75 contention, 64/66/69/70 its refusals, 255 ssh). 0 also when no shape
# changed (one line says so). 64: an argument this script does not take, or a DIR that is no bloomery tree. 70: recipes.py
# failed or printed a line this script cannot run (a word outside [A-Za-z0-9_,.:=/-]), or a box run that ended before its last
# shape line.
set -uo pipefail

say() { echo "$*" >&2; }

# on_box CMD…: run on the box, in the remote tree (tools/box.sh's directory): every shape, the first red rc.
on_box() {
  local n=$# i=0 rc=0 r t0 red=""
  for c in "$@"; do
    i=$((i + 1))
    echo "round-check: [$i/$n] $c"
    t0=$SECONDS
    # Word splitting is the point: each shape is one command line whose words the Mac side checked.
    # shellcheck disable=SC2086
    $c
    r=$?
    echo "round-check: [$i/$n] rc=$r wall=$((SECONDS - t0))s"
    if [ "$r" != 0 ]; then
      red="$red $i"
      [ "$rc" != 0 ] || rc=$r
    fi
  done
  red=${red# }
  echo "round-check: on-box done shapes=$n red=${red:-none} rc=$rc"
  return "$rc"
}

# A shape's words: what combos prints, and nothing a shell would read as more than a word.
SHAPE_RE='^cargo (check|clippy)( [A-Za-z0-9_,.:=/-]+)+$'

ledger_dir() { echo "${BLOOMERY_ROUND_LEDGER_DIR:-$HOME/.cache/bloomery}"; }

# predict MODE SHAPES: the median per-shape wall of the last five runs of MODE, times SHAPES; empty when none ran yet.
predict() {
  local f
  f="$(ledger_dir)/round-check-times.tsv"
  [ -f "$f" ] || return 0
  awk -F'\t' -v m="$1" -v n="$2" '$3 == m && $4 > 0 { r[++k] = $5 / $4 }
    END { if (!k) exit; s = (k > 5 ? k - 4 : 1); c = 0; for (i = s; i <= k; i++) v[++c] = r[i]
          for (i = 1; i <= c; i++) for (j = i + 1; j <= c; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
          med = (c % 2 ? v[(c + 1) / 2] : (v[c / 2] + v[c / 2 + 1]) / 2); printf "%d", med * n + 0.5 }' "$f"
}

main() {
  local tree base=origin/main mode=check fresh=0 dry=0
  tree=$(cd "$(dirname "$0")/.." && pwd)
  while [ $# -gt 0 ]; do
    case $1 in
      --tree) [ $# -ge 2 ] || { say "round-check.sh: --tree needs a directory"; return 64; }; tree=$2; shift 2 ;;
      --base) [ $# -ge 2 ] || { say "round-check.sh: --base needs a revision"; return 64; }; base=$2; shift 2 ;;
      --clippy) mode=clippy; shift ;;
      --fresh) fresh=1; shift ;;
      --dry-run) dry=1; shift ;;
      *) say "round-check.sh: '$1' is not an argument (tools/round-check.sh [--tree DIR] [--base SPEC] [--clippy] [--fresh] [--dry-run] | --self-test)"; return 64 ;;
    esac
  done
  [ -f "$tree/tools/box.sh" ] && [ -f "$tree/tools/recipes.py" ] ||
    { say "round-check.sh: $tree is no bloomery tree (no tools/box.sh and tools/recipes.py)"; return 64; }
  tree=$(cd "$tree" && pwd)
  local track remote
  track=$(basename "$tree")
  if [ -n "${BLOOMERY_REMOTE:-}" ]; then
    remote=$BLOOMERY_REMOTE
  elif [ "$track" = bloomery ]; then
    say "round-check.sh: $tree is the main tree, whose remote directory ~/repo/bloomery is the lead's; a round checks in its own (set BLOOMERY_REMOTE to name one)"
    return 64
  else
    # shellcheck disable=SC2088 # the box's shell expands the '~'
    remote="~/repo/bloomery-${track#bloomery-}"
  fi
  local led_dir ledger out rc=0
  led_dir=$(ledger_dir)
  ledger="$led_dir/round-check-$mode.tsv"
  local largs=(--base "$base")
  [ "$fresh" = 1 ] || largs+=(--ledger "$ledger")
  out=$(python3 "$tree/tools/recipes.py" combos "${largs[@]}") || rc=$?
  if [ "$rc" != 0 ]; then
    say "round-check.sh: tools/recipes.py combos ${largs[*]} failed (rc $rc)"
    return 70
  fi
  local -a cmds=() keys=() labels=()
  local kind label cmd key total=""
  while IFS=$'\t' read -r kind label cmd key; do
    case $kind in
      combo)
        [ "$mode" = check ] || cmd="cargo clippy ${cmd#cargo check }"
        if ! [[ $cmd =~ $SHAPE_RE ]] || ! [[ $key =~ ^[0-9a-f]{64}$ ]]; then
          say "round-check.sh: combos printed a shape this script cannot run as words, or with no input key: $cmd ${key:-<no key>}"
          return 70
        fi
        cmds+=("$cmd") keys+=("$key") labels+=("$label")
        ;;
      total) total=$label ;;
    esac
  done <<< "$out"
  echo "round-check: $tree -> $remote, base $base, mode $mode: ${total:-no total line}"
  if [ ${#cmds[@]} = 0 ]; then
    echo "round-check: no shape to check (no input of any build shape changed since $base, or each is green in $ledger)"
    return 0
  fi
  local pred
  pred=$(predict "$mode" "${#cmds[@]}")
  [ -n "$pred" ] && pred="$pred s" || pred="unknown (no earlier run)"
  echo "round-check: ${#cmds[@]} shape(s), predicted $pred; ledger ${ledger}$([ "$fresh" = 1 ] && echo ' (--fresh: not read)')"
  local i q=""
  for i in "${!cmds[@]}"; do
    echo "round-check: shape $((i + 1)): ${labels[$i]}"
    q="$q '${cmds[$i]}'"
  done
  [ "$dry" = 0 ] || { echo "round-check: dry run, the box command: bash tools/round-check.sh --on-box$q"; return 0; }
  local logdir log t0 brc=0
  logdir="$tree/target/round-check"
  mkdir -p "$logdir" || { say "round-check.sh: cannot make $logdir"; return 70; }
  log=$(mktemp "$logdir/run.XXXXXX") || { say "round-check.sh: cannot make a log under $logdir"; return 70; }
  t0=$SECONDS
  BLOOMERY_REMOTE=$remote bash "$tree/tools/box.sh" "bash tools/round-check.sh --on-box$q" 2>&1 | tee "$log"
  brc=${PIPESTATUS[0]}
  local wall=$((SECONDS - t0))
  if ! grep -q '^round-check: on-box done ' "$log"; then
    echo "round-check: the box run ended rc $brc before its summary line (log $log)"
    [ "$brc" != 0 ] || brc=70
    return "$brc"
  fi
  local stamp n
  stamp=$(date +%Y-%m-%dT%H:%M:%S%z)
  mkdir -p "$led_dir"
  n=0
  for i in "${!cmds[@]}"; do
    if grep -q "^round-check: \[$((i + 1))/${#cmds[@]}\] rc=0 " "$log"; then
      printf '%s\tgreen\t%s\t%s\n' "${keys[$i]}" "$stamp" "$tree" >> "$ledger"
      n=$((n + 1))
    fi
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$stamp" "$tree" "$mode" "${#cmds[@]}" "$wall" "$brc" >> "$led_dir/round-check-times.tsv"
  BLOOMERY_REMOTE=$remote bash "$tree/tools/lock-back.sh" || say "round-check.sh: tools/lock-back.sh failed: Cargo.lock in $tree may be behind the box's"
  echo "round-check: rc=$brc wall=${wall}s shapes=${#cmds[@]} green=$n (recorded in $ledger) log=$log"
  return "$brc"
}

self_test() {
  local d fails=0 n=0 out rc here
  here=$(cd "$(dirname "$0")" && pwd)
  d=$(mktemp -d "${TMPDIR:-/tmp}/round-check-test.XXXXXX") || return 70
  trap 'rm -rf "$d"' RETURN
  # A fake tree: this script, a stub recipes.py that prints $d/combos.out, a stub box.sh that records its environment and
  # argument and runs the on-box loop against a stub cargo, and a stub lock-back.sh.
  mk_tree() {
    local t=$1
    mkdir -p "$t/tools" "$t/bin"
    cp "$here/round-check.sh" "$t/tools/round-check.sh"
    cat > "$t/tools/recipes.py" << 'PY'
import os, sys
open(os.environ["RC_T"] + "/recipes.args", "w").write(" ".join(sys.argv[1:]) + "\n")
sys.stdout.write(open(os.environ["RC_T"] + "/combos.out").read())
sys.exit(int(os.environ.get("RC_COMBOS_RC", "0")))
PY
    cat > "$t/tools/box.sh" << 'SH'
#!/usr/bin/env bash
printf '%s\n' "$BLOOMERY_REMOTE" > "$RC_T/box.remote"
printf '%s\n' "$*" > "$RC_T/box.cmd"
[ -z "${RC_BOX_RC:-}" ] || { echo "box.sh: stub refusal"; exit "$RC_BOX_RC"; }
cd "$(dirname "$0")/.." && PATH="$PWD/bin:$PATH" bash -c "$*"
SH
    cat > "$t/tools/lock-back.sh" << 'SH'
#!/usr/bin/env bash
echo lock-back >> "$RC_T/lockback.calls"
SH
    # The stub cargo: a shape whose package is `red` fails with 101; every call is logged.
    cat > "$t/bin/cargo" << 'SH'
#!/usr/bin/env bash
echo "$*" >> "$RC_T/cargo.calls"
case " $* " in *" -p red "*) echo "error[E0277]: stub"; exit 101 ;; esac
exit 0
SH
    chmod +x "$t/bin/cargo"
  }
  local k1 k2 k3
  k1=$(printf 'a%.0s' {1..64}) k2=$(printf 'b%.0s' {1..64}) k3=$(printf 'c%.0s' {1..64})
  local tab=$'\t'
  # case NAME EXPECT_RC PATTERN [ENV=V…] -- ARGS…: run the copy in $d/<tree> and match its output.
  check() {
    local name=$1 want=$2 pat=$3
    shift 3
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    n=$((n + 1))
    rm -f "$d"/box.* "$d"/cargo.calls "$d"/lockback.calls
    rc=0
    out=$(env -i PATH="/usr/bin:/bin:$(dirname "$(command -v python3)")" HOME="$d/home" RC_T="$d" BLOOMERY_ROUND_LEDGER_DIR="$d/led" \
      ${envs[@]+"${envs[@]}"} bash "$d/bloomery-tr/tools/round-check.sh" "$@" 2>&1) || rc=$?
    if [ "$rc" != "$want" ] || ! grep -qE -- "$pat" <<< "$out"; then
      echo "round-check self-test FAIL: $name (rc $rc, want $want; pattern '$pat')" >&2
      printf '%s\n' "$out" | sed 's/^/  | /' >&2
      fails=$((fails + 1))
    fi
  }
  mk_tree "$d/bloomery-tr"
  mk_tree "$d/bloomery"
  mkdir -p "$d/home"
  printf 'combo%sgood build: 1 targets%scargo check -p good --features a,b --lib%s%s\ncombo%sred test: 1 targets%scargo check -p red --profile test --bin x%s%s\nskip%sother%sno input file changed\ntotal%s2 of 9 commands to run\n' \
    "$tab" "$tab" "$tab" "$k1" "$tab" "$tab" "$tab" "$k2" "$tab" "$tab" "$tab" > "$d/combos.out"

  check 'an unknown argument is refused' 64 "'--bogus' is not an argument" -- --bogus
  check 'a tree with no tools is refused' 64 'is no bloomery tree' -- --tree "$d/home"
  check 'the main tree is refused without a remote' 64 'is the main tree' -- --tree "$d/bloomery"
  check 'the main tree with a remote named runs' 101 'rc=101' BLOOMERY_REMOTE='~/repo/x' -- --tree "$d/bloomery"
  # Every shape runs, the red one does not stop the green one after it, and the rc is the red one's.
  check 'a red shape: its rc, every shape run' 101 "round-check: \[2/2\] rc=101" --
  [ "$(wc -l < "$d/cargo.calls" | tr -d ' ')" = 2 ] || { echo "round-check self-test FAIL: both shapes ran (cargo calls: $(cat "$d/cargo.calls" 2>/dev/null))" >&2; fails=$((fails + 1)); }
  [ "$(cat "$d/box.remote")" = '~/repo/bloomery-tr' ] || { echo "round-check self-test FAIL: the remote is ~/repo/bloomery-<track> (got $(cat "$d/box.remote"))" >&2; fails=$((fails + 1)); }
  [ -f "$d/lockback.calls" ] || { echo "round-check self-test FAIL: lock-back runs after a box run" >&2; fails=$((fails + 1)); }
  # The green shape is in the ledger, the red one is not; the next run hands combos the ledger.
  grep -q "^$k1${tab}green$tab" "$d/led/round-check-check.tsv" 2>/dev/null && ! grep -q "^$k2" "$d/led/round-check-check.tsv" ||
    { echo "round-check self-test FAIL: the ledger holds the green shape alone" >&2; fails=$((fails + 1)); }
  grep -q -- "--ledger $d/led/round-check-check.tsv" "$d/recipes.args" ||
    { echo "round-check self-test FAIL: combos reads the ledger ($(cat "$d/recipes.args"))" >&2; fails=$((fails + 1)); }
  check '--fresh reads no ledger' 101 'not read' -- --fresh
  grep -q -- '--ledger' "$d/recipes.args" && { echo "round-check self-test FAIL: --fresh passed a ledger" >&2; fails=$((fails + 1)); }
  check 'a predicted wall after one run' 101 'shape\(s\), predicted [0-9]+ s; ledger' --
  check 'BLOOMERY_REMOTE is passed through' 101 'rc=101' BLOOMERY_REMOTE='~/repo/other' --
  [ "$(cat "$d/box.remote")" = '~/repo/other' ] || { echo "round-check self-test FAIL: BLOOMERY_REMOTE passes to box.sh" >&2; fails=$((fails + 1)); }
  check '--clippy runs each shape as clippy' 101 'ledger .*/round-check-clippy\.tsv' -- --clippy
  grep -q '^clippy -p good' "$d/cargo.calls" || { echo "round-check self-test FAIL: --clippy ran cargo clippy" >&2; fails=$((fails + 1)); }
  check 'a box.sh refusal is its rc, no lock-back' 75 'ended rc 75 before its summary line' RC_BOX_RC=75 --
  [ -f "$d/lockback.calls" ] && { echo "round-check self-test FAIL: lock-back after a box run that did not run" >&2; fails=$((fails + 1)); }
  check '--dry-run prints the box command and runs nothing' 0 "dry run, the box command: bash tools/round-check.sh --on-box 'cargo check -p good" -- --dry-run
  [ -f "$d/box.cmd" ] && { echo "round-check self-test FAIL: --dry-run called box.sh" >&2; fails=$((fails + 1)); }
  check 'a combos failure is 70' 70 'combos .* failed \(rc 3\)' RC_COMBOS_RC=3 --
  printf 'total%s0 of 9 commands to run\n' "$tab" > "$d/combos.out"
  check 'no shape changed: 0, nothing sent' 0 'no shape to check' --
  [ -f "$d/box.cmd" ] && { echo "round-check self-test FAIL: no shape still called box.sh" >&2; fails=$((fails + 1)); }
  printf 'combo%sx%scargo check -p x --features "a;rm"%s%s\n' "$tab" "$tab" "$tab" "$k3" > "$d/combos.out"
  check 'a shape with a word the shell would read is 70' 70 'cannot run as words' --
  printf 'combo%sx%scargo check -p x --lib\n' "$tab" "$tab" > "$d/combos.out"
  check 'a shape with no key is 70' 70 'with no input key' --
  # The on-box loop by itself: the first red rc, every shape, the summary line.
  n=$((n + 1))
  rc=0
  out=$(cd "$d/bloomery-tr" && PATH="$d/bloomery-tr/bin:$PATH" RC_T="$d" bash tools/round-check.sh --on-box 'cargo check -p red --lib' 'cargo check -p good --lib' 2>&1) || rc=$?
  if [ "$rc" != 101 ] || ! grep -q '^round-check: on-box done shapes=2 red=1 rc=101$' <<< "$out"; then
    echo "round-check self-test FAIL: --on-box (rc $rc): $out" >&2
    fails=$((fails + 1))
  fi
  if [ "$fails" != 0 ]; then
    echo "round-check self-test: $fails of $n cases failed" >&2
    return 1
  fi
  echo "round-check self-test: $n cases ok"
}

case ${1:-} in
  --self-test) self_test; exit $? ;;
  --on-box) shift; on_box "$@"; exit $? ;;
esac
main "$@"
exit $?
