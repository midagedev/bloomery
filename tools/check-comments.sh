#!/usr/bin/env bash
# 주석 규약 점검(맥, 빌드 없음). AGENTS.md Conventions: 주석에 이슈 번호·날짜를 쓰지 않는다 —
# 이력은 rig-log와 커밋 메시지의 것이다. 예외는 재핀 귀속 한 줄 `PIN(YYYY-MM-DD):`.
# 대상은 crates/*/src와 crates/*/tests 전부다. 목록이 아니라 제외로 적어서 새 크레이트가 조용히 빠지지 않게 한다. 제외는
# 워크스페이스 밖의 재현기 oxide-ice-unroll뿐이다.
# The comments are read by tools/check-comment-only.py --comments (one lexer: `//` and `/* */`, never a `//`
# inside a string). Two rules, each outside a `PIN(` line:
#   crates/*/src and crates/*/tests: no issue number (MUL-N) and no date in a comment;
#   crates/*/src: no measured value in a comment — a number with a time, rate or bandwidth unit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
dirs=()
for d in crates/*/src crates/*/tests; do
  [ -d "$d" ] || continue
  case "$(basename "$(dirname "$d")")" in
    oxide-ice-unroll) ;;
    *) dirs+=("$d") ;;
  esac
done
files=()
while IFS= read -r f; do files+=("$f"); done < <(find "${dirs[@]}" -name '*.rs' | LC_ALL=C sort)
rc=0
comments=$(python3 tools/check-comment-only.py --comments "${files[@]}") || rc=$?
if [ "$rc" != 0 ]; then
  echo "check-comments: the comment reader failed (rc $rc, above) — no verdict" >&2
  exit "$rc"
fi
# g: grep whose "no line" (1) is an answer and whose error (2) ends the check by name.
g() {
  local rc=0
  grep "$@" || rc=$?
  [ "$rc" -le 1 ] || { echo "check-comments: grep $* failed (rc $rc) — no verdict" >&2; return "$rc"; }
}
# Each line is `path:line:text`; a rule matches the text after the line number.
text() { g -E "^[^:]+:[0-9]+:.*($1)" | g -v 'PIN('; }
fail=0
bad=$(printf '%s\n' "$comments" | text 'MUL-[0-9]+|20[0-9]{2}-[0-9]{2}-[0-9]{2}')
if [ -n "$bad" ]; then
  n=$(printf '%s\n' "$bad" | wc -l | tr -d ' ')
  printf '%s\n' "$bad" | head -40 >&2
  echo "check-comments: $n comment line(s) carry an issue number or a date — history belongs in rig-log (AGENTS.md Conventions)" >&2
  fail=1
fi
unit='(^|[^0-9A-Za-z_])[0-9][0-9,.]*[[:space:]]?(ms|µs|μs|us|ns|GB/s|GiB/s|tok/s|t/s|MB/s)([^0-9A-Za-z_]|$)'
meas=$(printf '%s\n' "$comments" | g -E '^crates/[^/]+/src/' | text "$unit")
if [ -n "$meas" ]; then
  n=$(printf '%s\n' "$meas" | wc -l | tr -d ' ')
  printf '%s\n' "$meas" | head -40 >&2
  echo "check-comments: $n comment line(s) in crates/*/src carry a measured value — the number belongs in rig-log (AGENTS.md Conventions: measured ms/GB/s/tok/s)" >&2
  fail=1
fi
[ "$fail" = 0 ] || exit 1
echo "check-comments: ok (${#files[@]} files, $(printf '%s\n' "$comments" | wc -l | tr -d ' ') comment lines)"
