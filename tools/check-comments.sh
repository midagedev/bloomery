#!/usr/bin/env bash
# 주석 규약 점검(맥, grep뿐). AGENTS.md Conventions: 주석에 이슈 번호·날짜를 쓰지 않는다 —
# 이력은 rig-log와 커밋 메시지의 것이다. 예외는 재핀 귀속 한 줄 `PIN(YYYY-MM-DD):`.
# 대상은 엔진 크레이트의 src/ (스테이지 0 크레이트 q3k-*는 MUL-10/11의 몫이라 제외).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bad=$(grep -rnE '//.*(MUL-[0-9]+|20[0-9]{2}-[0-9]{2}-[0-9]{2})' \
        "$ROOT"/crates/{gguf,threads,qdot,model}/src --include='*.rs' | grep -v 'PIN(' || true)
if [ -n "$bad" ]; then
  n=$(printf '%s\n' "$bad" | wc -l | tr -d ' ')
  printf '%s\n' "$bad" | head -40 >&2
  echo "check-comments: $n comment line(s) carry an issue number or a date — history belongs in rig-log (AGENTS.md Conventions)" >&2
  exit 1
fi
echo "check-comments: ok"
