#!/usr/bin/env bash
# The release's carry check: every fix tools/release/carry.tsv lists must be an ancestor of the commit being
# released. `just release-build` runs it on the Mac, where .git is (tools/box.sh syncs none), before it touches
# the box.
#
# Usage: tools/release/carry.sh [REV]     the rows against REV (default HEAD)
#        tools/release/carry.sh --self-test   (a temp git repo and carry.tsv; check-recipes runs it)
#
# A row is `commit<TAB>name<TAB>why` (carry.tsv's header says what a row means and when one is added). It is
# carried when `git merge-base --is-ancestor <commit> REV`. Every refusal is named, each on its own line, in
# one report before the exit:
#   - a row whose commit is `pending` (a fix known and not landed) or not an ancestor of REV: `lacks`;
#   - a commit this repository does not hold (unknown, ambiguous, or not a commit): its own refusal, never
#     "not an ancestor";
#   - a malformed row, by its line number: not three tab-separated columns, an empty column, or a commit that is
#     neither 7 to 40 hex digits nor `pending` (a branch name is an ancestor of its own tip and would pass);
#   - a carry.tsv that is missing or holds no row (an emptied list checks nothing).
# The refusals carry `release-build:`, the recipe that calls this script.
#
# Exit: 0 every row carried; 1 a refusal; 64 an argument this script does not take, or a REV that is not a
# commit.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TSV=tools/release/carry.tsv

# carry <rev>: the report on stdout (all carried) or stderr (refusals), the exit code above.
carry() {
  local rev=$1 full short line lineno=0 rows=0 bad=0 tabs commit name why rest rc msgs=''
  full=$(git -C "$ROOT" rev-parse --verify --quiet "$rev^{commit}") || {
    echo "carry.sh: $rev is not a commit in $ROOT" >&2
    return 64
  }
  short=$(git -C "$ROOT" rev-parse --short=12 "$full")
  [ -f "$ROOT/$TSV" ] || {
    echo "release-build: $TSV is missing" >&2
    return 1
  }
  refuse() { msgs="$msgs$1"$'\n'; bad=$((bad + 1)); }
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    case $line in *[![:space:]]*) ;; *) continue ;; esac
    case $line in '#'*) continue ;; esac
    rows=$((rows + 1))
    # Columns are counted on the tabs: `read` with a tab IFS merges adjacent tabs and would lose an empty name.
    tabs=${line//[^$'\t']/}
    if [ "${#tabs}" != 2 ]; then
      refuse "release-build: $TSV:$lineno: not 3 tab-separated columns (commit, name, why), found $((${#tabs} + 1))"
      continue
    fi
    commit=${line%%$'\t'*}
    rest=${line#*$'\t'}
    name=${rest%%$'\t'*}
    why=${rest#*$'\t'}
    case $name in *[![:space:]]*) ;; *) refuse "release-build: $TSV:$lineno: the name is empty"; continue ;; esac
    case $why in *[![:space:]]*) ;; *) refuse "release-build: $TSV:$lineno: the why of $name is empty"; continue ;; esac
    if [ "$commit" = pending ]; then
      refuse "release-build: $rev $short lacks $name (pending): $why"
      continue
    fi
    case $commit in
      *[!0-9a-fA-F]* | '') refuse "release-build: $TSV:$lineno: the commit of $name is '$commit', neither a hash nor pending"; continue ;;
    esac
    if [ "${#commit}" -lt 7 ] || [ "${#commit}" -gt 40 ]; then
      refuse "release-build: $TSV:$lineno: the commit of $name is '$commit', not 7 to 40 hex digits"
      continue
    fi
    if ! git -C "$ROOT" cat-file -e "$commit^{commit}" 2> /dev/null; then
      refuse "release-build: $TSV:$lineno: $name ($commit) is not a commit this repository holds (unknown, ambiguous or not a commit)"
      continue
    fi
    git -C "$ROOT" merge-base --is-ancestor "$commit" "$full"
    rc=$?
    case $rc in
      0) ;;
      1) refuse "release-build: $rev $short lacks $name ($commit): $why" ;;
      *) refuse "release-build: $TSV:$lineno: git merge-base --is-ancestor $commit $rev failed (rc $rc)" ;;
    esac
  done < "$ROOT/$TSV"
  if [ "$rows" = 0 ]; then
    echo "release-build: $TSV holds no row: an empty list checks nothing" >&2
    return 1
  fi
  if [ "$bad" != 0 ]; then
    printf '%s' "$msgs" >&2
    echo "release-build: $rev $short is refused: $bad of $rows rows of $TSV" >&2
    return 1
  fi
  echo "release-build: $rev $short carries all $rows rows of $TSV"
}

self_test() {
  local t me fails=0 cases=0 out rc h1 h2 s1 main g unknown=0123456789abcdef0123456789abcdef01234567
  me=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE
  t=$(mktemp -d) || { echo "carry.sh: self-test: no temporary directory" >&2; return 70; }
  SELF_TMP=$t
  trap 'rm -rf "$SELF_TMP"' EXIT
  mkdir -p "$t/tools/release"
  cp "$me" "$t/tools/release/carry.sh"
  g="git -C $t -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null -c commit.gpgsign=false"
  $g init -q
  $g commit -q --allow-empty -m first
  h1=$($g rev-parse HEAD)
  s1=$($g rev-parse --short=7 HEAD)
  main=$($g rev-parse --abbrev-ref HEAD)
  # The fix on a branch that HEAD does not carry: the shape of the release candidate that missed a fix.
  $g checkout -q -b later
  $g commit -q --allow-empty -m second
  h2=$($g rev-parse HEAD)
  $g checkout -q "$main"
  rows() { printf '%b' "$1" > "$t/$TSV"; }
  run() { # run <args...>: the copy's carry.sh on the fixture; sets out (both streams) and rc
    out=$(cd "$t" && bash tools/release/carry.sh "$@" 2>&1)
    rc=$?
  }
  expect() { # expect <case> <want rc> <substring>... : every substring in out
    local name=$1 want=$2
    shift 2
    cases=$((cases + 1))
    local ok=1 s
    [ "$rc" = "$want" ] || ok=0
    for s in "$@"; do grep -qF -- "$s" <<< "$out" || ok=0; done
    if [ "$ok" = 0 ]; then
      echo "carry self-test: $name: rc $rc (want $want), want in the output: $*" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  absent() { # absent <case> <substring>: the substring is not in out
    cases=$((cases + 1))
    if grep -qF -- "$2" <<< "$out"; then
      echo "carry self-test: $1: '$2' must not be in the output:" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  local pre='# a comment\n\n   \n'

  rows "${pre}$h1\tone\tthe first fix\n$s1\ttwo\ta short hash\n"
  run
  expect carried 0 "release-build: HEAD ${h1:0:12} carries all 2 rows of $TSV"

  rows "${pre}$h1\tone\tthe first fix\n$h2\ttwo\tthe fix on the branch\n"
  run
  expect lacks 1 "release-build: HEAD ${h1:0:12} lacks two ($h2): the fix on the branch" "1 of 2 rows"
  absent lacks-carried-row 'lacks one'
  run later
  expect rev-argument 0 "release-build: later ${h2:0:12} carries all 2 rows"

  rows "${pre}$h1\tone\tthe first fix\npending\tsoon\tnot landed yet\n"
  run
  expect pending 1 "release-build: HEAD ${h1:0:12} lacks soon (pending): not landed yet" "1 of 2 rows"

  rows "${pre}$h1\tone\tthe first fix\n$unknown\tghost\ta hash nobody has\n"
  run
  expect unknown-hash 1 "$TSV:5: ghost ($unknown) is not a commit this repository holds"
  absent unknown-is-not-lacks 'lacks ghost'

  rows "${pre}$h1\tone\n"
  run
  expect two-columns 1 "$TSV:4: not 3 tab-separated columns (commit, name, why), found 2"
  rows "${pre}$h1\tone\twhy\textra\n"
  run
  expect four-columns 1 "$TSV:4: not 3 tab-separated columns (commit, name, why), found 4"
  rows "${pre}$h1\t\twhy\n"
  run
  expect empty-name 1 "$TSV:4: the name is empty"
  rows "${pre}$h1\tone\t\n"
  run
  expect empty-why 1 "$TSV:4: the why of one is empty"
  rows "${pre}$main\tone\twhy\n"
  run
  expect branch-name 1 "$TSV:4: the commit of one is '$main', neither a hash nor pending"
  rows "${pre}${h1:0:5}\tone\twhy\n"
  run
  expect short-hash 1 "$TSV:4: the commit of one is '${h1:0:5}', not 7 to 40 hex digits"

  rows "${pre}$h2\ttwo\tlacked\nbad line\n$unknown\tghost\tunknown\npending\tsoon\tnot landed\n"
  run
  expect all-named 1 "lacks two ($h2): lacked" "$TSV:5: not 3 tab-separated columns (commit, name, why), found 1" "ghost ($unknown) is not a commit" \
    "lacks soon (pending): not landed" "4 of 4 rows"

  rows "# only a comment\n\n"
  run
  expect no-rows 1 "$TSV holds no row"
  rm -f "$t/$TSV"
  run
  expect no-file 1 "$TSV is missing"
  rows "$h1\tone\twhy\n"
  run nowhere
  expect rev-not-commit 64 'nowhere is not a commit'
  run one two
  expect two-args 64 'usage'
  run --nope
  expect unknown-flag 64 'usage'

  if [ "$fails" = 0 ]; then
    echo "carry: self-test ok ($cases cases)"
  else
    echo "carry: self-test $fails of $cases cases failed" >&2
    return 1
  fi
}

case ${1:-} in
  --self-test) [ $# = 1 ] && { self_test; exit $?; } ;;
  -*) ;;
  *) [ $# -le 1 ] && { carry "${1:-HEAD}"; exit $?; } ;;
esac
echo "usage: tools/release/carry.sh [REV] | --self-test" >&2
exit 64
