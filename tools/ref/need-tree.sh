#!/usr/bin/env bash
# shellcheck shell=bash
# The refusal of a script that needs a live reference tree (an ik_llama.cpp or llama.cpp checkout and its build) when the tree
# is not there. Sourced; `bash tools/ref/need-tree.sh --self-test` runs the cases (check-recipes runs it). A recipe that cannot
# source it runs `bash tools/ref/need-tree.sh --check [--rc N] WHO VAR PATH [REMEDY]`: the same call, line and exit code.
#
# No gate reads a live tree: a gate runs on the code and on frozen reference sets (crates/refset). What still reads a tree is a
# writer of those sets (build-*, dump-*) and a timing runner's reference arms (decode-measure.sh, ab-decode.sh with
# BLOOMERY_AB_IK=1, the depth runners). Each is an opt-in that works only on a machine that keeps the tree, so its absence is
# one named refusal at the top of the script, before a lease or a build, and never a failed `g++` or a missing binary three
# minutes in.
#
#   need_tree [--rc N] WHO VAR PATH [REMEDY]
#
# Returns 0 when PATH exists (a file or a directory: the caller names the one it reads). Otherwise prints one line that begins
# WHO, names VAR and PATH, says what to do, and exits N (default 66: the missing-input code of cb-dump.sh and
# build-k3-quant-tree.sh; the timing runners pass 2, the code their own "no llama-bench at" refusals use). It exits the caller,
# as oracle_tree does, so call it at the top level of a script, not in a `$(...)`.
need_tree() {
  local rc=66
  if [ "${1:-}" = --rc ]; then
    [ $# -ge 2 ] || { echo "need_tree: --rc takes a code" >&2; exit 64; }
    rc=$2
    shift 2
  fi
  [ $# -ge 3 ] || { echo "need_tree: usage: need_tree [--rc N] WHO VAR PATH [REMEDY]" >&2; exit 64; }
  local who=$1 var=$2 path=$3 remedy=${4:-}
  [ -e "$path" ] && return 0
  echo "$who: no live reference tree: $var=$path is not there; no gate reads one (frozen sets only), this script is the opt-in that" \
    "does${remedy:+; $remedy}" >&2
  exit "$rc"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if [ "${1:-}" = --check ]; then
    shift
    need_tree "$@"
    exit 0
  fi
  [ "${1:-}" = --self-test ] || { echo "need-tree.sh is sourced; --check runs one call, --self-test its cases" >&2; exit 64; }
  fails=0 checks=0
  check() {
    checks=$((checks + 1))
    if [ "$2" != "$3" ]; then
      echo "FAIL $1: got [$2] want [$3]" >&2
      fails=$((fails + 1))
    fi
  }
  t=$(mktemp -d "${TMPDIR:-/tmp}/need-tree-self-test.XXXXXX")
  touch "$t/llama-bench"
  # A present file and a present directory pass and print nothing.
  out=$(need_tree who IKBIN "$t/llama-bench" 2>&1) && r=0 || r=$?
  check file-present "$r|$out" '0|'
  out=$(need_tree who IK "$t" 2>&1) && r=0 || r=$?
  check dir-present "$r|$out" '0|'
  # An absent path exits 66 with one line that begins WHO and names the variable and the path.
  out=$(need_tree who IKBIN "$t/nope/llama-bench" 2>&1) && r=0 || r=$?
  check absent-rc "$r" 66
  check absent-lines "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" 1
  case $out in
    "who: no live reference tree: IKBIN=$t/nope/llama-bench is not there;"*) check absent-text ok ok ;;
    *) check absent-text "$out" 'who: no live reference tree: IKBIN=...' ;;
  esac
  # The remedy is the line's tail; --rc moves the code; a bare call is a usage error.
  out=$(need_tree --rc 2 who IK "$t/nope" 'set IK_ARMS=0 to measure ours alone' 2>&1) && r=0 || r=$?
  check rc-2 "$r" 2
  case $out in *"does; set IK_ARMS=0 to measure ours alone") check remedy ok ok ;; *) check remedy "$out" '...does; set IK_ARMS=0...' ;; esac
  out=$(need_tree who 2>&1) && r=0 || r=$?
  check usage "$r" 64
  # The command-line form is the same call: present passes, absent is the one line and 66, the remedy rides along.
  out=$(bash "${BASH_SOURCE[0]}" --check who IK "$t" 2>&1) && r=0 || r=$?
  check cli-present "$r|$out" '0|'
  out=$(bash "${BASH_SOURCE[0]}" --check lab-x K3QUANT "$t/nope/llama-quantize" 'just build-k3-quant-tree builds it' 2>&1) && r=0 || r=$?
  check cli-absent-rc "$r" 66
  case $out in "lab-x: no live reference tree: K3QUANT=$t/nope/llama-quantize is not there;"*"does; just build-k3-quant-tree builds it") check cli-absent-text ok ok ;; *) check cli-absent-text "$out" 'lab-x: no live reference tree: ...' ;; esac
  rm -rf "$t"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($checks checks, $fails failures)"
  [ "$fails" = 0 ]
  exit
fi
