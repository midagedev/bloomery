#!/usr/bin/env bash
# 주석 규약 점검(맥, grep뿐). AGENTS.md Conventions: 주석에 이슈 번호·날짜를 쓰지 않는다 —
# 이력은 rig-log와 커밋 메시지의 것이다. 예외는 재핀 귀속 한 줄 `PIN(YYYY-MM-DD):`.
# 대상은 crates/*/src 전부다. 목록이 아니라 제외로 적어서 새 크레이트가 조용히 빠지지 않게 한다. 제외는
# 스테이지 0 크레이트 q3k-gemv·q3k-cpu·gpu-spike(MUL-10/11의 몫)와 워크스페이스 밖의 재현기 oxide-ice-unroll뿐이다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
dirs=()
for d in "$ROOT"/crates/*/src; do
  case "$(basename "$(dirname "$d")")" in
    q3k-gemv | q3k-cpu | gpu-spike | oxide-ice-unroll) ;;
    *) dirs+=("$d") ;;
  esac
done
bad=$(grep -rnE '//.*(MUL-[0-9]+|20[0-9]{2}-[0-9]{2}-[0-9]{2})' \
        "${dirs[@]}" --include='*.rs' | grep -v 'PIN(' || true)
if [ -n "$bad" ]; then
  n=$(printf '%s\n' "$bad" | wc -l | tr -d ' ')
  printf '%s\n' "$bad" | head -40 >&2
  echo "check-comments: $n comment line(s) carry an issue number or a date — history belongs in rig-log (AGENTS.md Conventions)" >&2
  exit 1
fi
echo "check-comments: ok"
