#!/usr/bin/env bash
# GPU 게이트 러너 — 박스에서, box.sh가 들어간 원격 디렉터리에서 돈다. 레시피가 방금 지은
# target/release/<이름>을 카드의 게이트 락 아래에서 상한과 함께 돌리고, 그 종료 코드를
# 그대로 돌려준다. 이름 뒤의 인자는 바이너리에 그대로 붙는다.
#   ./tools/box.sh 'cargo oxide build … --bin gate_p3 && bash tools/gpu-gate.sh gate_p3'
#
# 막는 실패: 매달린 GPU 게이트 하나가 게이트 락을 무기한 쥐는 것. 락은 3090 게이트를 한 줄로 세우는 자리라,
# 한 게이트가 서면 그 락을 기다리는 트랙이 전부 선다. CPU 게이트의 900초 상한(tools/gate.sh)과 같은 값을
# 쓰고, 넘으면 빨간 게이트로 끝난다. 락과 상한의 소유자는 이 스크립트 하나다 — 레시피에 락 경로가 직접
# 나오면 check-recipes가 빨강이다.
#
# 종료 코드: 게이트의 것 그대로. 124 = 상한에서 TERM으로 끝남, 137 = TERM을 무시해 --kill-after의 KILL로
# 끝남, 75 = 30분 안에 락을 못 잡음(경쟁이지 게이트 실패가 아니다), 64 = 사용법, 2 = 바이너리 없음,
# 69 = 게이트 락 파일을 열 수 없음(박스의 root 셸이 아님).
# 환경: BLOOMERY_GATE_BOUND(초, 기본 900 — tools/gate.sh와 같은 레버), BLOOMERY_GATE_CARD(아래 카드 고르기),
# BLOOMERY_BOX_CARD(box.sh가 넘기는 카드 선택 — 아래).
set -uo pipefail
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: gpu-gate.sh <target/release binary> [args...]" >&2; exit 64; }
shift
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/gate-bound.sh"
gate_bound gpu-gate.sh || exit $?
EXE=./target/release/$NAME
[ -x "$EXE" ] || { echo "gpu-gate.sh: no $EXE — the recipe builds it before calling this" >&2; exit 2; }
# 카드 고르기. 락은 카드마다 하나: 3090은 예전 경로 그대로(돌고 있는 트랙의 옛 사본이 그 경로를 잡는다),
# A6000은 새 파일. BLOOMERY_GATE_CARD=3090(기본 — 예전과 같다) | a6000 | any. any는 A6000을 먼저 본다 —
# V4.1 `--place gate` 게이트는 3090에서만 돌 수 있으니 떠도는 게이트가 3090을 비워 두는 편이 낫다. A6000은
# 타이밍 임대(/root/bloomery-cpu.lock)가 잡혀 있거나 그 카드에 컴퓨트 프로세스가 있으면 건너뛴다. 임대 탐침은
# lease-probe.sh의 lease_free(공유 잠금) 하나다 — 대기 중 5초마다 도는 이 탐침이 배타 잠금이면 다른 탐침과 부딪혀
# 빈 임대를 잡힌 것으로 읽고, 테스트할 수 없는 임대도 비었다고 읽지 않는다.
# shellcheck source=tools/ref/lease-probe.sh
source "${BASH_SOURCE[0]%/*}/ref/lease-probe.sh"
# box.sh가 보여 준 카드가 락을 정한다. BLOOMERY_BOX_CARD는 box.sh가 박스 쪽에 넘기는 그 선택이다(3090 = env
# 파일의 핀, a6000, both = 두 카드). box.sh를 거치지 않은 실행은 핀 그대로로 읽는다. box.sh가 BLOOMERY_CARD=a6000|both로
# 카드를 골랐으면 BLOOMERY_GATE_CARD 없이 그 카드의 락을 잡고 — both는 두 락을 함께 잡거나 하나도 안 잡는다 — 다른 카드를
# 이름 대면 거절한다(64). 두 카드를 보는 실행이 3090 락 하나만 쥐면 다른 트랙의 any 게이트가 비어 보이는 A6000에 올라가
# 같은 카드 메모리를 나눠 쓴다.
BOXC=${BLOOMERY_BOX_CARD:-3090}
case "$BOXC" in
  3090) CARD=${BLOOMERY_GATE_CARD:-3090} ;;
  a6000 | both)
    CARD=${BLOOMERY_GATE_CARD:-$BOXC}
    [ "$CARD" = "$BOXC" ] || {
      echo "gpu-gate.sh: box.sh put $([ "$BOXC" = both ] && echo 'both cards' || echo 'the A6000') in view (BLOOMERY_CARD=$BOXC), and BLOOMERY_GATE_CARD=$CARD names another lock — leave it unset or set it to $BOXC" >&2
      exit 64
    } ;;
  *) echo "gpu-gate.sh: BLOOMERY_BOX_CARD is 3090, a6000 or both (box.sh sets it), got '$BOXC'" >&2; exit 64 ;;
esac
case "$CARD" in
  3090 | a6000 | any) ;;
  both)
    [ "$BOXC" = both ] || {
      echo "gpu-gate.sh: BLOOMERY_GATE_CARD=both holds both cards' locks for a run that sees both — run it under box.sh's BLOOMERY_CARD=both" >&2
      exit 64
    } ;;
  *) echo "gpu-gate.sh: BLOOMERY_GATE_CARD is 3090, a6000, any or both, got '$CARD'" >&2; exit 64 ;;
esac
uuid_of() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "$1" | cut -d, -f1 | head -1; }
a6000_idle() {
  local u; u=$(uuid_of A6000); [ -n "$u" ] || return 1
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"
}
# 락 파일을 못 여는 셸(박스의 root가 아니거나 박스가 아님)은 경쟁이 아니다 — 여기서 이름을 대고 끝난다.
for lock in 9:/root/bloomery-gate.lock 8:/root/bloomery-gate-a6000.lock; do
  eval "exec ${lock%%:*}>${lock#*:}" 2> /dev/null || { echo "gpu-gate.sh: cannot open ${lock#*:} — the gate runs on the box, as root" >&2; exit 69; }
done
take_a6000() { flock -n 8 || return 1; if [ "$CARD" = any ] && ! { lease_free && a6000_idle; }; then flock -u 8; return 1; fi; GOT=a6000; }
take_3090() { flock -n 9 || return 1; GOT=3090; }
GOT=
if [ "$CARD" = both ]; then
  # 3090 락을 먼저, 그것을 쥔 채 A6000 락을. 다른 실행은 락을 하나만 쥐므로 이 순서로는 교착이 없고, 막혀 기다리는
  # flock은 해제 순간 깨어나 5초마다 도는 한 카드 실행들의 폴링보다 먼저 잡는다(둘이 함께 빌 때만 잡는 폴링은 게이트가
  # 이어지는 동안 굶는다).
  start=$SECONDS
  if flock -w 1800 9; then
    left=$((1800 - (SECONDS - start)))
    if flock -w "$((left > 0 ? left : 1))" 8; then GOT=both; else flock -u 9; fi
  fi
else
  for ((waited = 0; waited <= 1800; waited += 5)); do
    case "$CARD" in
      3090) take_3090 ;;
      a6000) take_a6000 ;;
      any) take_a6000 || take_3090 ;;
    esac
    [ -n "$GOT" ] && break
    sleep 5
  done
fi
if [ -z "$GOT" ]; then
  echo "gpu-gate.sh: no gate lock ($CARD) was free within 30 min — contention, not a red gate" >&2
  exit 75
fi
if [ "$GOT" = a6000 ] || [ "$CARD" = any ]; then
  U=$(uuid_of "$([ "$GOT" = a6000 ] && echo A6000 || echo 3090)")
  [ -n "$U" ] || { echo "gpu-gate.sh: the $GOT lookup failed" >&2; exit 75; }
  export CUDA_VISIBLE_DEVICES=$U
fi
echo "gpu-gate.sh: $NAME on $([ "$GOT" = both ] && echo 'both cards, both gate locks' || echo "the $GOT") (asked $CARD)" >&2
timeout --kill-after=10 "$BOUND" "$EXE" "$@"
rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GPU GATE TIMED OUT: $NAME after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GPU GATE RED: $NAME (exit $rc)" >&2
fi
exit "$rc"
