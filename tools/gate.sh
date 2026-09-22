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
set -uo pipefail
BOUND=${BLOOMERY_GATE_BOUND:-900}
RUN=(cargo test)
if [ "${1:-}" = --oxide ]; then
  shift
  RUN=(cargo oxide test --arch sm_86 --)
fi
timeout --kill-after=10 "$BOUND" "${RUN[@]}" "$@"
rc=$?
# 124 = timeout이 TERM으로 끝냄, 137 = TERM을 무시해 --kill-after의 KILL로 끝냄.
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GATE TIMED OUT after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GATE RED (exit $rc)" >&2
fi
exit "$rc"
