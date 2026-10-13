#!/usr/bin/env bash
# Whether the commits after the release batch block the tag. A commit that moves no shipped binary does not: its owning
# gates run after the user-facing pass (or beside it, on a card the pass does not hold), never before the tag.
#
# Usage: tools/release/late-commit.sh BASE HEAD [--scan-logs BASE_LOG HEAD_LOG]... [--files-from FILE]
#        tools/release/late-commit.sh --self-test   (replays 0.3.1's cut from this repository's history)
#
# BASE is the commit the release batch ran on, HEAD the commit the tag would name; the changed files are
# `git diff --name-only --no-renames BASE HEAD` (--files-from FILE reads the list from FILE, one path a line, and is how
# --self-test tries a list no commit has). Two things must hold, and the script says 0 only when both do:
#   1. No changed file is inside the shipped closure (`python3 tools/recipes.py ships`): the bins of
#      tools/release/build.sh built with its --features, their module trees (`mod`, `#[path]`, `include_str!`), the
#      bloomery-gpu-gates lib, every workspace crate in their dependency graph at those features (lib tree, manifest,
#      build script), Cargo.lock, rust-toolchain.toml, .cargo/, tools/release/ and the scripts build.sh names. The
#      closure is read at HEAD and at BASE, so a deleted or moved-away file counts. A path pattern ("gate_* bins, tests,
#      tools, docs") does not decide it: crates/gpu-gates/src/bin/shared/serve_seats/ and crates/gpu-gates/src/*.rs look
#      like gate code and are in the shipped binary.
#   2. The ptx-scan pair of each shipped bin, BASE against HEAD, is identical (`python3 tools/recipes.py ship-scan`):
#      the check that no kernel moved, which the closure alone cannot see if its model has a hole. The pair needs the
#      box (it builds both commits and runs ptxas): without --scan-logs this script prints the exact commands
#      (`NARROW_SHIP=1 tools/narrow-scan.sh BASE HEAD --out DIR`, which adds the shipped bins at the release
#      features to its scan set, then this script again with the two logs of each bin) and exits 1.
#
# Exit: 0 does not block the tag; 1 otherwise, and the first line says why: `BLOCKS` (a changed file inside the closure,
# named with its reason, or a scan pair that is not identical) or `UNPROVEN` (the files hold and the scan pair is not
# given, or the tool cannot decide); 64 an argument this script does not take, or a BASE or HEAD that is not a commit;
# 70 --self-test without the history of the cut.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USAGE="usage: tools/release/late-commit.sh BASE HEAD [--scan-logs BASE_LOG HEAD_LOG]... [--files-from FILE] | --self-test"

die() { # die <rc> <message>
  echo "late-commit: $2" >&2
  exit "$1"
}

# late_commit <base rev> <head rev> <files-from or ''> <scan log>...: the verdict on stdout, the exit code above.
late_commit() {
  local base_rev=$1 head_rev=$2 files_from=$3 base head f nfiles=0 out err rc n shown=0 cmd_out
  shift 3
  local -a files=() logs=("$@") bins=()
  base=$(git -C "$ROOT" rev-parse --verify --quiet "$base_rev^{commit}") || die 64 "$base_rev is not a commit in $ROOT"
  head=$(git -C "$ROOT" rev-parse --verify --quiet "$head_rev^{commit}") || die 64 "$head_rev is not a commit in $ROOT"
  if [ -n "$files_from" ]; then
    [ -r "$files_from" ] || die 64 "--files-from: $files_from is not a readable file"
    while IFS= read -r f || [ -n "$f" ]; do [ -z "$f" ] || files[${#files[@]}]=$f; done < "$files_from"
  else
    while IFS= read -r f; do [ -z "$f" ] || files[${#files[@]}]=$f; done < <(git -C "$ROOT" -c core.quotepath=false diff --name-only --no-renames "$base" "$head")
  fi
  nfiles=${#files[@]}
  local range="${base:0:12}..${head:0:12}"
  if [ "$nfiles" = 0 ]; then
    echo "late-commit: $range changes no file: does not block the tag"
    return 0
  fi

  err=$(mktemp "${TMPDIR:-/tmp}/late-commit.XXXXXX") || die 70 "no temporary file"
  out=$(cd "$ROOT" && python3 tools/recipes.py ships --rev "$head" --base "$base" "${files[@]}" 2> "$err")
  rc=$?
  case $rc in
    0) ;;
    1)
      n=$(printf '%s\n' "$out" | grep -c .)
      echo "late-commit: BLOCKS the tag: $n of $nfiles files changed in $range are inside the shipped closure:"
      printf '%s\n' "$out" | head -n 20 | while IFS=$'\t' read -r f why; do echo "  $f — $why"; done
      [ "$n" -le 20 ] || echo "  … and $((n - 20)) more"
      rm -f "$err"
      return 1
      ;;
    *)
      echo "late-commit: UNPROVEN: tools/recipes.py ships could not decide (rc $rc): $(tr '\n' ' ' < "$err")" >&2
      rm -f "$err"
      return 1
      ;;
  esac
  rm -f "$err"

  if [ "${#logs[@]}" = 0 ]; then
    local dir="$HOME/.cache/bloomery/late-commit/${base:0:8}-${head:0:8}" b
    while IFS=$'\t' read -r b _; do [ -z "$b" ] || bins[${#bins[@]}]=$b; done < <(cd "$ROOT" && python3 tools/recipes.py ship-bins --rev "$head" 2> /dev/null)
    echo "late-commit: UNPROVEN: none of the $nfiles files changed in $range is inside the shipped closure, and the scan pair is not given."
    echo "  On the box (BASE and HEAD are built and scanned there, --no-jit, no card):"
    echo "    NARROW_SHIP=1 tools/narrow-scan.sh ${base:0:12} ${head:0:12} --out $dir"
    echo "  then, with the logs of each shipped bin:"
    cmd_out="    tools/release/late-commit.sh ${base:0:12} ${head:0:12}"
    for b in ${bins[@]+"${bins[@]}"}; do cmd_out="$cmd_out --scan-logs $dir/base/$b.log $dir/head/$b.log"; done
    echo "$cmd_out"
    return 1
  fi

  out=$(cd "$ROOT" && python3 tools/recipes.py ship-scan --rev "$head" "${logs[@]}" 2>&1)
  rc=$?
  case $rc in
    0)
      printf '%s\n' "$out" | sed 's/^/  scan /'
      echo "late-commit: none of the $nfiles files changed in $range is inside the shipped closure, and the scan pair is identical: does not block the tag"
      return 0
      ;;
    1)
      echo "late-commit: BLOCKS the tag: the scan pair of a shipped bin is not identical:"
      printf '%s\n' "$out" | sed 's/^/  /'
      return 1
      ;;
    *)
      echo "late-commit: BLOCKS the tag: the scan pair is refused:"
      printf '%s\n' "$out" | sed 's/^/  /'
      return 1
      ;;
  esac
}

# --- self-test: 0.3.1's cut, replayed from this repository's history (no checkout: every closure is a git archive) ---------
# h = d571d068, 87b79b4c, 5ae948b0 (gate bins only), g = 5d210be6 (crates/gpu/src/arch/qwen3moe/body.rs) and 77b7daf5 (a gate
# bin and a tsv). The scan pair is a fixture pair (the replay has no box): identical for the verdict-0 cases, moved for one.

mk_scan() { # mk_scan <path> <bin> <digest of the entry beta>
  {
    echo "ptx-scan: mod1 bundle=bloomery-gpu bytes=100"
    echo "ptx-scan: mod2 bundle=bloomery-gpu-deepseek41 bytes=100"
    echo "ptx-scan: mod3 bundle=bloomery-gpu-vision bytes=100"
    echo "ptx-scan bin=target/release/$2 section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=3 jit=skipped(no-jit)"
    echo "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local"
    printf '%-24s 256 no 0 0 0 0 12 0 0 6 skipped(no-jit) skipped(no-jit)\n' alpha beta
    echo "ptx-scan-md5: method=decl1"
    echo "alpha 11111111111111111111111111111111 41"
    echo "beta $3 41"
  } > "$1"
}

self_test() {
  local t cases=0 fails=0 c me="$ROOT/tools/release/late-commit.sh" out rc
  for c in d571d068 87b79b4c 5ae948b0 5d210be6 77b7daf5; do
    git -C "$ROOT" cat-file -e "$c^{commit}" 2> /dev/null || {
      echo "late-commit: self-test: commit $c, one of 0.3.1's cut, is not in this repository's history: fetch the full history" >&2
      return 70
    }
  done
  t=$(mktemp -d "${TMPDIR:-/tmp}/late-commit-test.XXXXXX") || die 70 "self-test: no temporary directory"
  local b2="22222222222222222222222222222222" b3="33333333333333333333333333333333"
  mk_scan "$t/base.log" bloomery-serve "$b2"
  mk_scan "$t/same.log" bloomery-serve "$b2"
  mk_scan "$t/moved.log" bloomery-serve "$b3"
  mk_scan "$t/gate.log" gate_kquant "$b2"
  printf '%s\n' crates/gpu-gates/src/bin/shared/serve_seats/qwen3.rs > "$t/seats.txt"
  printf '%s\n' crates/gpu-gates/src/bin/gate_glm5next_serve.rs tools/defaults-twins.tsv justfile > "$t/gateonly.txt"

  # replay <label> <want rc> <must appear (or -)> <args...>: prints the verdict line of each case
  replay() {
    local label=$1 want=$2 must=$3 first
    shift 3
    out=$(bash "$me" "$@" 2>&1)
    rc=$?
    first=$(printf '%s\n' "$out" | grep -m1 '^late-commit:' | cut -c1-150)
    cases=$((cases + 1))
    if [ "$rc" = "$want" ] && { [ "$must" = - ] || grep -qF -- "$must" <<< "$out"; }; then
      echo "ok   $label: rc $rc — ${first:-no verdict line}"
    else
      fails=$((fails + 1))
      echo "FAIL $label: rc $rc (want $want${must:+, and in the output: $must}) — ${first:-no verdict line}"
      printf '%s\n' "$out" | sed 's/^/       /' | head -n 12
    fi
  }
  local pair="--scan-logs $t/base.log $t/same.log"
  # h: gate bins only
  replay "d571d068 gate_qwen38_residency, gate_qwen38_serve" 0 "does not block the tag" d571d068^ d571d068 $pair
  replay "87b79b4c gate_mimo2_e2e" 0 "does not block the tag" 87b79b4c^ 87b79b4c $pair
  replay "5ae948b0 gate_mimo2_serve" 0 "does not block the tag" 5ae948b0^ 5ae948b0 $pair
  replay "h as one range 87b79b4c^..5ae948b0" 0 "does not block the tag" 87b79b4c^ 5ae948b0 $pair
  # g: a kernel-carrying lib file, then a gate bin and a tsv
  replay "5d210be6 crates/gpu/src/arch/qwen3moe/body.rs" 1 "crates/gpu/src/arch/qwen3moe/body.rs" 5d210be6^ 5d210be6 $pair
  replay "77b7daf5 gate_glm5next_serve, defaults-twins.tsv" 0 "does not block the tag" 77b7daf5^ 77b7daf5 $pair
  replay "the cut as one range 77b7daf5^..5ae948b0" 1 "crates/gpu/src/arch/qwen3moe/body.rs" 77b7daf5^ 5ae948b0 $pair
  # a synthetic list: serve_seats/ looks like gate code and is in the binary; gate-only paths are not
  replay "serve_seats/qwen3.rs (synthetic list)" 1 "crates/gpu-gates/src/bin/shared/serve_seats/qwen3.rs" 5ae948b0 5ae948b0 --files-from "$t/seats.txt" $pair
  replay "gate bin, tsv and justfile (synthetic list)" 0 "does not block the tag" 5ae948b0 5ae948b0 --files-from "$t/gateonly.txt" $pair
  # the scan half
  replay "5ae948b0 with a moved scan pair" 1 "BLOCKS the tag: the scan pair of a shipped bin is not identical" 5ae948b0^ 5ae948b0 --scan-logs "$t/base.log" "$t/moved.log"
  replay "5ae948b0 with the scan of a gate bin" 1 "not of a shipped bin" 5ae948b0^ 5ae948b0 --scan-logs "$t/gate.log" "$t/gate.log"
  replay "5ae948b0 with no scan pair" 1 "UNPROVEN" 5ae948b0^ 5ae948b0
  replay "5ae948b0 with no scan pair: the command" 1 "NARROW_SHIP=1 tools/narrow-scan.sh" 5ae948b0^ 5ae948b0
  replay "no change at all" 0 "changes no file" 5ae948b0 5ae948b0
  # usage
  replay "no argument" 64 "usage"
  replay "an unknown option" 64 "usage" 5ae948b0^ 5ae948b0 --nope
  replay "one log of a pair" 64 "usage" 5ae948b0^ 5ae948b0 --scan-logs "$t/base.log"
  replay "a BASE that is not a commit" 64 "is not a commit" no-such-rev 5ae948b0
  rm -rf "$t"
  if [ "$fails" = 0 ]; then
    echo "late-commit: self-test ok ($cases cases)"
  else
    echo "late-commit: self-test: $fails of $cases cases failed" >&2
    return 1
  fi
}

[ "${1:-}" = --self-test ] && { [ $# = 1 ] || die 64 "$USAGE"; self_test; exit $?; }
[ $# -ge 2 ] || { echo "$USAGE" >&2; exit 64; }
base_rev=$1
head_rev=$2
shift 2
files_from=
scan_logs=()
while [ $# -gt 0 ]; do
  case $1 in
    --scan-logs)
      [ $# -ge 3 ] || { echo "$USAGE" >&2; exit 64; }
      scan_logs[${#scan_logs[@]}]=$2
      scan_logs[${#scan_logs[@]}]=$3
      shift 3
      ;;
    --files-from)
      [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 64; }
      files_from=$2
      shift 2
      ;;
    *) echo "$USAGE" >&2; exit 64 ;;
  esac
done
late_commit "$base_rev" "$head_rev" "$files_from" ${scan_logs[@]+"${scan_logs[@]}"}
exit $?
