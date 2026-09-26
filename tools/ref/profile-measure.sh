#!/usr/bin/env bash
# 디코드 스텝의 시간 귀속을, 박스 전역 임대 안에서, 행마다 증인을 남기고 잰다.
#
# 프로파일 표는 손으로 돌리면 안 되는 부류다. 옆에서 빌드 하나만 돌아도 site 간
# **비율**이 흔들리고, 이 표의 쓸모는 정확히 그 비율이다. 그래서 decode-measure.sh 와
# 같은 임대·같은 증인 형식을 쓴다.
#
# 레벨 둘을 연달아 찍는다. 레벨 1은 호출당 Instant 한 쌍이라 세금이 없고 site별
# 배분의 원본이다. 레벨 2는 행마다 두 번 불러 단계(활성 양자화 / 가중치 디퀀트 / 내적)를
# 가르는 대신 절대 시간에 타이머 세금이 붙는다 — **비율만 발견이고 절대값은 아니다**.
# 둘을 나란히 두는 이유가 그것이다: 레벨 1이 "어느 site 가 큰가"를, 레벨 2가 "그 site
# 안에서 어느 단계가 큰가"를 답한다. 한 표에 섞으면 세금이 배분까지 오염시킨다.
set -euo pipefail
# 모델·데이터 디렉터리 기본값(BLOOMERY_REF_MODEL·BLOOMERY_DATA 오버라이드는 그대로 받는다)은 빌드
# 스크립트와 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
TOKENS=${BLOOMERY_DECODE_TOKENS:-$REF_TOKENS}
# BLOOMERY_PROFILE_DEPTH=d 이면 깊이 d 의 표다: depth-decode.sh 가 쓰는 lcg_prompt 를 그대로 프리필해 캐시를
# 그 깊이로 채운 뒤 잰다. 어텐션 항은 캐시된 키 수에 선형이라 기본 프롬프트(깊이 6)의 표에서는 스텝의
# 몇 % 로만 보이고, 깊이 4096 에서야 절반 가까이 된다 — 깊이를 안 적은 어텐션 배분은 수치가 아니다.
if [ -n "${BLOOMERY_PROFILE_DEPTH:-}" ]; then
  TOKENS=$(lcg_prompt "$BLOOMERY_PROFILE_DEPTH")
fi
# 기본이 2 스텝인 이유: 이 표가 답하는 질문은 배분이고 배분은 스텝 하나로 정해진다.
# 스텝을 늘리면 임대만 길어진다. 평탄성은 decode-measure.sh 가 재는 다른 질문이다.
N=${BLOOMERY_DECODE_N:-2}
BIN=${BLOOMERY_DECODE_BIN:-$DECODE_BIN}
[ -x "$BIN" ] || { echo "no decode binary at $BIN — run: just build-decode" >&2; exit 2; }

WITNESS=(head loadavg pressure-cpu pressure-io gpus lock-holder model busiest)

lease_take

for lvl in ${BLOOMERY_PROFILE_LEVELS:-1 2}; do
  echo
  echo "=== BLOOMERY_PROFILE=$lvl ==="
  witness "pre-level$lvl"
  lease_bounded "$LEASE_ARM_BOUND" env BLOOMERY_PROFILE="$lvl" "$BIN" -m "$MODEL" --tokens "$TOKENS" -n "$N"
  witness "post-level$lvl"
done
