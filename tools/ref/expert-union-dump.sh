#!/usr/bin/env bash
# 프롬프트 파일 전체를 탐욕 디코드하며 층별 전문가 id와 생성 토큰을 떨어뜨린다 (박스에서 실행).
# 시간 측정이 아니지만 코어를 다 쓰므로 임대 안에서 돈다. 분석은 tools/expert-union.py.
set -uo pipefail
# 모델 기본값(BLOOMERY_REF_MODEL 오버라이드는 그대로 받는다)은 빌드 스크립트와 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
N=${BLOOMERY_DECODE_N:-128}
OUT=${1:?out dir}
BIN=target/release/bloomery-decode
[ -x "$BIN" ] || { echo "no decode binary — run just build-decode" >&2; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"
exec 9>/root/bloomery-cpu.lock
flock -w 1800 9 || { echo "[lease] timed out" >&2; exit 75; }
grep -v '^#' "$REF_PROMPTS" | while IFS=$'\t' read -r id _text toks; do
  [ -n "$toks" ] || continue
  BLOOMERY_EXPERT_LOG="$OUT/$id.experts" "$BIN" -m "$MODEL" --tokens "$toks" -n "$N" > "$OUT/$id.out" 2>&1 \
    || { echo "prompt $id failed" >&2; exit 1; }
  echo "$toks" > "$OUT/$id.prompt"
  grep '^generated' "$OUT/$id.out" > "$OUT/$id.gen"
done
ls "$OUT"/*.gen | wc -l
