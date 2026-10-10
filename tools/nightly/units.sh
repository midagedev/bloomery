#!/usr/bin/env bash
# The one unit runner: a library, sourced by tools/nightly/run.sh (the nightly host run) and tools/contrib-check.sh (a
# contributor's off-box check). A unit is a `name<TAB>command` line of `python3 tools/recipes.py host-nightly`, or any
# command a caller names; this file runs one and reads its outcome, and a caller decides what to do with it.
#
#   run_unit CMD LOG BOUND [quiet]
#     runs `bash -c CMD` with stdin from /dev/null under `timeout --kill-after=10 BOUND+30` (the 30 s leave room for the
#     unit's own tools/gate.sh bound to speak first); the output goes to LOG, and to stdout too unless the fourth word is
#     `quiet`. Sets:
#       UNIT_RC     the command's exit code (124/137: the bound ran out)
#       UNIT_SECS   its wall time in whole seconds
#       UNIT_KIND   `cargo` when CMD runs tools/gate.sh, else `script`
#       UNIT_P, UNIT_F, UNIT_I   the sums of the log's `test result:` passed, failed and ignored counts
#     UNIT_TIMEOUT names the bounding command (default `timeout`); a host without it (macOS has no GNU timeout) sets it to
#     the empty string, and the unit then runs unbounded: the caller says so.
#   unit_failure NAME RC LOG BOUND
#     prints what a red unit's `failed` file holds: one `NAME: <test>` line per libtest `... FAILED` line, or else one
#     `NAME: <why>` line (timeout, a filter that matched no test, a build failure, the rc) and the first few red lines (a rustfmt
#     `Diff in` line included).
#
# The two callers stay in step by sharing this code: a change to how a unit's counts or failures are read lands here once.

# shellcheck disable=SC2034  # the UNIT_* variables are this library's result interface, read by the sourcing script
UNIT_TIMEOUT=${UNIT_TIMEOUT-timeout}

run_unit() {
  local cmd=$1 log=$2 bound=$3 mode=${4:-} t0=$SECONDS
  local -a bounded=()
  if [ -n "$UNIT_TIMEOUT" ]; then bounded=("$UNIT_TIMEOUT" --kill-after=10 "$((bound + 30))"); fi
  if [ "$mode" = quiet ]; then
    BLOOMERY_GATE_BOUND=$bound ${bounded[@]+"${bounded[@]}"} bash -c "$cmd" < /dev/null > "$log" 2>&1
    UNIT_RC=$?
  else
    BLOOMERY_GATE_BOUND=$bound ${bounded[@]+"${bounded[@]}"} bash -c "$cmd" < /dev/null 2>&1 | tee "$log"
    UNIT_RC=${PIPESTATUS[0]}
  fi
  UNIT_SECS=$((SECONDS - t0))
  UNIT_KIND=script
  case $cmd in *gate.sh*) UNIT_KIND=cargo ;; esac
  read -r UNIT_P UNIT_F UNIT_I < <(awk '/^test result: / { for (k = 2; k <= NF; k++) { if ($k == "passed;") p += $(k - 1); if ($k == "failed;") f += $(k - 1); if ($k == "ignored;") g += $(k - 1) } } END { print p + 0, f + 0, g + 0 }' "$log")
}

unit_failure() {
  local name=$1 rc=$2 log=$3 bound=$4 named why
  named=$(sed -n 's/^test \(.*\) \.\.\. FAILED$/\1/p' "$log")
  if [ -n "$named" ]; then
    printf '%s\n' "$named" | sed "s/^/$name: /"
  else
    case $rc in
      124 | 137) why="timeout after ${bound}s (rc $rc)" ;;
      78) why="a test filter matched no test (rc 78)" ;;
      *) if grep -q 'could not compile\|^error\(\[E[0-9]*\]\)\?:' "$log"; then why="build failure (rc $rc)"; else why="rc $rc"; fi ;;
    esac
    echo "$name: $why"
    # a script's red lines sit anywhere in its output, not in its last 80: name the first few
    grep -E 'FAIL|[Ff]ailed|[Ee]rror|^Diff in' "$log" | grep -v '^ok ' | head -n 8 | cut -c1-220 | sed 's/^/    | /'
  fi
}
