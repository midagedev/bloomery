#!/usr/bin/env bash
# 게이트 러너 — 박스에서, box.sh가 들어간 원격 디렉터리에서 돈다. 인자는 `cargo test` 뒤에 그대로 붙는다.
#   ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test mt -- --ignored --nocapture'
# 첫 인자가 --oxide면 디바이스 크레이트용으로 `cargo oxide test --arch sm_86 --` 뒤에 붙는다(plain cargo는
# 디바이스 크레이트를 빌드하지 못한다 — AGENTS.md 「Never」).
#   ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu --release --lib'
#
# 막는 실패 둘:
#  1. 매달린 게이트(2026-09-20 q_nope2 무한루프가 gate-mt를 한 시간 넘게 붙잡음) — 900초 상한,
#     넘으면 빨간 게이트로 끝난다.
#  2. 삼켜진 종료 코드 — 상한을 처음 넣은 ef9e579의 레시피는 `timeout … cargo test … || echo "TIMED OUT"`
#     꼴이었고, echo가 성공하므로 시험 실패든 타임아웃이든 레시피가 0으로 끝났다(같은 날 저녁 리뷰에서
#     발견; 평범한 시험 실패에도 "TIMED OUT"이 찍혔다). 종료 코드의 소유자는 이 스크립트 하나다:
#     cargo의 코드를 그대로 돌려주고, 타임아웃 문구는 타임아웃일 때만 찍는다.
# The bound is BLOOMERY_GATE_BOUND, parsed by tools/gate-bound.sh: a value it refuses ends this runner
# with 64 before cargo runs.
#
# A third failure it closes: a filter that matches nothing. libtest passes a run of 0 tests, so a
# call whose test-name filter names no test that exists (or none that runs under its ignore flags) is
# green and checks nothing. A call that carries a filter — an argument after its `--` that is not a
# flag, nor the value of an option in LIBTEST_VALUE below — and that cargo ends with 0 fails with 78
# when the `passed` counts of its `test result:` lines sum to 0 over the whole call (one target
# printing "running 0 tests" is normal: `--lib --test x -- f` names a test of one target). A call with
# no filter, or with `--list`, is not judged.
#
# Exit codes: cargo's own (0 green, 101 a test failed, …); 124 or 137 the bound ran out; 64 the bound
# was refused; 78 a filter matched no test that passed.
set -uo pipefail
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/gate-bound.sh"
gate_bound gate.sh || exit $?
RUN=(cargo test)
if [ "${1:-}" = --oxide ]; then
  shift
  RUN=(cargo oxide test --arch sm_86 --)
fi
# The libtest options that take a value. The one list: tools/recipes.py reads this line (by its
# `LIBTEST_VALUE=(` prefix, one line, one word per option) to tell a recipe's filters from values.
LIBTEST_VALUE=(--skip --test-threads --format --color --logfile -Z --shuffle-seed)
takes_value() {
  local o
  for o in "${LIBTEST_VALUE[@]}"; do
    [ "$1" = "$o" ] && return 0
  done
  return 1
}
FILTERS=()
LISTING=0
after=0
value=0
for a in "$@"; do
  if [ "$after" = 0 ]; then
    [ "$a" = -- ] && after=1
    continue
  fi
  if [ "$value" = 1 ]; then
    value=0
    continue
  fi
  if takes_value "$a"; then
    value=1
    continue
  fi
  case $a in
    --list) LISTING=1 ;;
    -*) ;;
    *) FILTERS+=("$a") ;;
  esac
done
if [ "${#FILTERS[@]}" = 0 ] || [ "$LISTING" = 1 ]; then
  timeout --kill-after=10 "$BOUND" "${RUN[@]}" "$@"
  rc=$?
else
  LOG=$(mktemp)
  trap 'rm -f "$LOG"' EXIT
  timeout --kill-after=10 "$BOUND" "${RUN[@]}" "$@" | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    passed=$(awk '/^test result: / { for (i = 2; i <= NF; i++) if ($i == "passed;") n += $(i - 1) } END { print n + 0 }' "$LOG")
    if [ "$passed" -eq 0 ]; then
      # tools/recipes.py's self-test reads the filters back from this line: keep "the filter … matched no test".
      echo "GATE RED (exit 78): the filter ${FILTERS[*]} matched no test — 0 passed over the whole call" >&2
      exit 78
    fi
  fi
fi
# 124 = timeout이 TERM으로 끝냄, 137 = TERM을 무시해 --kill-after의 KILL로 끝냄.
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GATE TIMED OUT after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GATE RED (exit $rc)" >&2
fi
exit "$rc"
