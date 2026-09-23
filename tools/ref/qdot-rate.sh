#!/usr/bin/env bash
# bloomery — qdot 커널률 러너. 박스에서 tools/box.sh 경유로 돈다(리드 전용: `just measure-qdot-rate`).
# ik 자신의 x4 커널(build-qdot-ref.sh가 짓는 *_rate 하네스 다섯)과 Rust `qdot-rate`를 같은 CPU 임대
# 안에서 번갈아 돈다. 둘 다 단일 스레드이고 형상이 같다(360,448행, 타입마다 같은 K). 한 코어에 고정하고,
# 라운드마다 팔 순서를 뒤집는다 — 고정 순서에서는 라운드의 첫 팔이 느리게 읽혔다(AGENTS.md, A/A).
# 인자: [라운드 수, 기본 3]. 환경: BLOOMERY_DATA(러너가 넘김), BLOOMERY_RATE_CORE(기본 2).
# 먼저 `just build-ref`(ik *_rate 하네스); qdot-rate는 레시피가 방금 지은 target/release의 것만 쓴다.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
ROUNDS=${1:-3}
CORE=${BLOOMERY_RATE_CORE:-2}
RUST=target/release/qdot-rate
IK_RATES="q4k_x4_rate q6k_x4_rate q5f0_rate q5f1_rate q5k_x4_rate"
case "$ROUNDS" in ''|*[!0-9]*|0) echo "qdot-rate.sh: rounds must be a positive integer, got '$ROUNDS'" >&2; exit 64 ;; esac
[ -x "$RUST" ] || { echo "no $RUST — run: just measure-qdot-rate (it builds qdot-rate first)" >&2; exit 2; }
for n in $IK_RATES; do
  [ -x "$BLOOMERY_DATA/bin/$n" ] || { echo "no $BLOOMERY_DATA/bin/$n — run: just build-ref" >&2; exit 2; }
done
WITNESS=(head loadavg pressure-cpu pressure-io gpus lock-holder core core-mhz)
ik_arm() { for n in $IK_RATES; do taskset -c "$CORE" "$BLOOMERY_DATA/bin/$n"; done; }
rust_arm() { taskset -c "$CORE" "$RUST"; }
lease_take
witness pre
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then
    echo "--- round $r: ik then rust"
    ik_arm
    rust_arm
  else
    echo "--- round $r: rust then ik"
    rust_arm
    ik_arm
  fi
done
witness post
