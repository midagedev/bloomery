#!/usr/bin/env bash
# CPU 거버너 A/B: 같은 임대 안에서 원래 거버너와 performance를 번갈아, bloomery와 ik 둘 다 잰다.
# 막는 실패: 거버너를 바꾼 채 죽는 것 — 모든 종료 경로에서 원래 값으로 복원한다(trap).
set -uo pipefail
# 모델·ik 트리·llama-bench 기본값(BLOOMERY_REF_MODEL·IK·IKBIN 오버라이드는 그대로 받는다)은 빌드 스크립트와
# 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
TOKENS=$REF_TOKENS
BIN=$DECODE_BIN
G=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
ORIG=$(cat "$G")
setgov() { for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$1" > "$f"; done; }
trap 'setgov "$ORIG"; echo "governor restored: $(cat $G)"' EXIT
ROUNDS=3
lease_take
echo "orig governor=$ORIG load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2)"
for r in $(seq "$ROUNDS"); do for gov in "$ORIG" performance; do
  setgov "$gov"; sleep 2
  b=$(lease_bounded "$LEASE_ARM_BOUND" "$BIN" -m "$MODEL" --tokens "$TOKENS" -n 96 2>&1 | grep -E 'decode steps' | sed 's/.*= //;s/ (.*//')
  i=$(lease_bounded "$LEASE_ARM_BOUND" env CUDA_VISIBLE_DEVICES= "$IKBIN" -m "$MODEL" -ngl 0 -t 32 -p 0 -n 32 -r 2 2>/dev/null | grep 'tg32' | awk -F'|' '{print $(NF-1)}')
  echo "r$r $gov | bloomery $b | ik $i | mhz $(awk '/MHz/{s+=$4;n++} END{printf "%.0f", s/n}' /proc/cpuinfo)"
done; done
