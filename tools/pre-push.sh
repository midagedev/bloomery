#!/usr/bin/env bash
# bloomery — the two checks a push request names, on one committed HEAD and at once: `just pre-review` (Shingler's
# reviewer, tools/review/local.py) and `just remote-static BASE` (mac-static on the second Mac, tools/mac-remote.sh).
#
#   tools/pre-push.sh [BASE]      BASE (default origin/main) is mac-static's scope; pre-review gets `--base BASE` only
#                                 when BASE is given (else its own default: the stacked parent, else origin/main)
#   tools/pre-push.sh --self-test the clean-tree rule, the parallel start, the lines and the exit code, against stubs
#
# A tree with uncommitted changes to tracked files is refused (64): mac-remote sends the worktree's content and
# pre-review reviews the committed HEAD, so only a clean tree makes them one SHA. The two run in parallel from the
# tree's root, each into its own log under ~/.cache/bloomery/pre-push/<sha8>/; when each ends it prints
# `pre-push: <check> rc <n> in <s> s: <its last line>`, and at the end the logs' directory.
# Exit: 0 both green; otherwise pre-review's rc when it is red, else mac-static's; 64 usage or a dirty tree.
# The failure it blocks: a push request whose two checks ran on two heads, or one after the other.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TREE=${PRE_PUSH_TREE:-$HERE}             # the tree checked: this script's, unless named (the self-test's scratch repo)
REVIEW=${PRE_PUSH_REVIEW:-"python3 $TREE/tools/review/local.py"}
STATIC=${PRE_PUSH_STATIC:-"$TREE/tools/mac-remote.sh"}  # mac-remote sends the tree it is in

say() { printf '%s\n' "$*" >&2; }

# run_one NAME LOG CMD...: the check's rc into LOG.rc, its output into LOG, then its one line.
run_one() {
  local name=$1 log=$2 t0=$SECONDS rc=0
  shift 2
  "$@" > "$log" 2>&1 < /dev/null || rc=$?
  echo "$rc" > "$log.rc"
  say "pre-push: $name rc $rc in $((SECONDS - t0)) s: $(tail -n 1 "$log" 2> /dev/null)"
}

pre_push() {
  local base=${1:-} sha dir rp rs rcp rcs
  [ $# -le 1 ] || { say "pre-push: takes at most one BASE"; return 64; }
  cd "$TREE"
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    say "pre-push: the tree has uncommitted changes to tracked files; commit them first (mac-static would check them, pre-review would not)"
    return 64
  fi
  sha=$(git rev-parse HEAD)
  dir=$HOME/.cache/bloomery/pre-push/${sha:0:8}
  mkdir -p "$dir"
  say "pre-push: pre-review and mac-static of ${sha:0:8} at once"
  # shellcheck disable=SC2086 # REVIEW and STATIC are command lines
  if [ -n "$base" ]; then run_one pre-review "$dir/pre-review.log" $REVIEW --base "$base" & else run_one pre-review "$dir/pre-review.log" $REVIEW & fi
  rp=$!
  # shellcheck disable=SC2086
  run_one mac-static "$dir/mac-static.log" $STATIC "${base:-origin/main}" &
  rs=$!
  wait "$rp" || true
  wait "$rs" || true
  rcp=$(cat "$dir/pre-review.log.rc" 2> /dev/null || echo 70)
  rcs=$(cat "$dir/mac-static.log.rc" 2> /dev/null || echo 70)
  say "pre-push: logs in $dir"
  [ "$rcp" = 0 ] || return "$rcp"
  return "$rcs"
}

self_test() {
  local d fails=0 rc out
  d=$(mktemp -d "${TMPDIR:-/tmp}/pre-push-test.XXXXXX")
  fail() { say "pre-push self-test FAIL: $*"; fails=$((fails + 1)); }
  git -C "$d" init -q -b main repo
  git -C "$d/repo" -c user.email=t@t -c user.name=t commit -q --allow-empty -m base
  echo a > "$d/repo/a"; git -C "$d/repo" add a
  git -C "$d/repo" -c user.email=t@t -c user.name=t commit -q -m a
  # each stub records its start, waits up to 5 s for the other's start (both run at once), prints, and ends with its rc
  printf '#!/bin/sh\ntouch %s/$1.started\ni=0; while [ ! -e %s/$2.started ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i+1)); done\n[ -e %s/$2.started ] && echo "$1 saw $2" || echo "$1 alone"\necho "$1 args: $*"\nexit $3\n' "$d" "$d" "$d" > "$d/stub"
  chmod +x "$d/stub"
  t() { PRE_PUSH_TREE=$d/repo HOME=$d/home PRE_PUSH_REVIEW="$d/stub review static $1" PRE_PUSH_STATIC="$d/stub static review $2" \
          bash "$HERE/tools/pre-push.sh" "${@:3}" 2>&1; }
  rc=0; out=$(t 0 0) || rc=$?
  [ "$rc" = 0 ] || fail "both green ended $rc: $out"
  case $out in *"pre-push: pre-review rc 0 in "*": review args: review static 0"*) ;; *) fail "pre-review's line: $out" ;; esac
  case $out in *"pre-push: mac-static rc 0 in "*"static args: static review 0 origin/main"*) ;; *) fail "mac-static's line, BASE origin/main: $out" ;; esac
  grep -q "review saw static" "$d/home/.cache/bloomery/pre-push/"*/pre-review.log && grep -q "static saw review" "$d/home/.cache/bloomery/pre-push/"*/mac-static.log ||
    fail "the two did not run at once: $(cat "$d/home/.cache/bloomery/pre-push/"*/*.log)"
  rm -f "$d"/*.started
  rc=0; out=$(t 0 3 HEAD~1) || rc=$?
  [ "$rc" = 3 ] || fail "a red mac-static (3) ended $rc: $out"
  case $out in *"review args: review static 0 --base HEAD~1"*"static args: static review 3 HEAD~1"*|*"static args: static review 3 HEAD~1"*"review args: review static 0 --base HEAD~1"*) ;;
    *) fail "a given BASE reaches both: $out" ;; esac
  rm -f "$d"/*.started
  rc=0; out=$(t 1 3) || rc=$?
  [ "$rc" = 1 ] || fail "pre-review red (1) and mac-static red (3) ended $rc, not pre-review's 1: $out"
  rm -f "$d"/*.started
  echo b >> "$d/repo/a"
  rc=0; out=$(t 0 0) || rc=$?
  [ "$rc" = 64 ] && [ ! -e "$d/review.started" ] && [ ! -e "$d/static.started" ] || fail "a dirty tree was not refused 64 before either ran: $rc $out"
  git -C "$d/repo" checkout -q -- a
  rc=0; out=$(t 0 0 a b) || rc=$?
  [ "$rc" = 64 ] || fail "two BASEs ended $rc, not 64"
  rm -rf "$d"
  [ "$fails" = 0 ] || { say "pre-push: self-test $fails failed"; return 1; }
  echo "pre-push: self-test ok"
}

case ${1:-} in
  --self-test) [ $# = 1 ] || { say "pre-push: --self-test takes nothing"; exit 64; }; self_test; exit $? ;;
  -*) say "pre-push: unknown option $1"; exit 64 ;;
esac
rc=0; pre_push "$@" || rc=$?
exit "$rc"
