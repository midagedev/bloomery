#!/usr/bin/env bash
# 주석 규약 점검(맥, 빌드 없음). AGENTS.md Conventions: 주석에 이슈 번호·날짜를 쓰지 않는다 —
# 이력은 rig-log와 커밋 메시지의 것이다. 예외는 재핀 귀속 한 줄 `PIN(YYYY-MM-DD):`.
# 대상은 crates/*/src와 crates/*/tests 전부다. 목록이 아니라 제외로 적어서 새 크레이트가 조용히 빠지지 않게 한다. 제외는
# 워크스페이스 밖의 재현기 oxide-ice-unroll뿐이다.
# The comments are read by tools/check-comment-only.py --comments (one lexer: `//` and `/* */`, never a `//`
# inside a string). Two rules, each outside a `PIN(` line:
#   crates/*/src and crates/*/tests: no issue number (MUL-N) and no date in a comment;
#   crates/*/src: no measured value in a comment — a number with a time, rate or bandwidth unit.
# A third rule reads every text file under tools/, comment or not: no `<doc>.md:<line>` citation (ranges and
# comma lists start the same way). A line number goes silently wrong the moment the doc is edited above it;
# cite a section heading plus a table row's first cell or a quoted phrase, or the rig-log
# `log/<date>.md#<anchor>` the value was measured in. An exception is a CITE_ALLOW entry, named with its reason.
# It sees the `.md:` form only: a doc named without its extension (`<report>:<line>`) is not caught.
# A fourth rule reads every file git sees (tracked, or untracked and not ignored): no word of a deleted
# feature (RETIRED_WORDS, a case-blind ERE spelled so this file cannot match it). An exception is a WORD_ALLOW entry: a whole file (`<path>||<reason>`) or one line
# (`<path>|<fixed text on the line>|<reason>`); an entry that matches nothing is an error.
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
# CITE_ALLOW: `<path>|<fixed text on the line>|<reason>`, one per allowed line; an entry that matches no
# citation is an error, so a stale exception cannot linger.
CITE_ALLOW=()
cites=$(g -rnIE '[A-Za-z0-9_./-]+\.md:[0-9]+' tools)
unmatched=()
allowed=0
for a in ${CITE_ALLOW[@]+"${CITE_ALLOW[@]}"}; do
  p=${a%%|*}; rest=${a#*|}; t=${rest%%|*}
  [ -n "$p" ] && [ -n "$t" ] && [ "$rest" != "$t" ] && [ -n "${rest#*|}" ] \
    || { echo "check-comments: CITE_ALLOW entry '$a' is not <path>|<text>|<reason>" >&2; exit 64; }
  hit=$(printf '%s\n' "$cites" | g -F -- "$t" | while IFS= read -r l; do case $l in ("$p:"*) printf '%s\n' "$l" ;; esac; done)
  if [ -z "$hit" ]; then unmatched+=("$a"); continue; fi
  allowed=$((allowed + $(printf '%s\n' "$hit" | wc -l)))
  cites=$(printf '%s\n' "$cites" | g -v -xF -- "$hit")
done
if [ "${#unmatched[@]}" -gt 0 ]; then
  printf 'check-comments: CITE_ALLOW entry matches no citation: %s\n' "${unmatched[@]}" >&2
  fail=1
fi
if [ -n "$cites" ]; then
  n=$(printf '%s\n' "$cites" | wc -l | tr -d ' ')
  printf '%s\n' "$cites" | head -40 >&2
  echo "check-comments: $n line(s) under tools/ cite a doc by line number — cite its section heading plus a table row's first cell or a quoted phrase, or a rig-log anchor" >&2
  fail=1
fi
# The router-frequency card list and its lever are deleted: the placement ranks no card set.
RETIRED_WORDS='h[o]t.?list|뜨[거]운 목록|핫 ?리[스]트'
WORD_ALLOW=(
  'crates/levers/src/registry.rs|name: "BLOOMERY_HOT_|the retired row refuses the name when it is set'
  'docs/plan-ledger.md||the lead owns the history in it'
  'docs/plan-triage.md||the lead owns the history in it'
)
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "check-comments: not a git work tree — the retired-word rule reads the tracked files; no verdict" >&2; exit 69; }
words=$(git grep --untracked -nIiE -- "$RETIRED_WORDS" || [ "$?" = 1 ])
wunmatched=()
wallowed=0
for a in "${WORD_ALLOW[@]}"; do
  p=${a%%|*}; rest=${a#*|}; t=${rest%%|*}
  [ -n "$p" ] && [ "$rest" != "$t" ] && [ -n "${rest#*|}" ] \
    || { echo "check-comments: WORD_ALLOW entry '$a' is not <path>|<text>|<reason>" >&2; exit 64; }
  hit=$(printf '%s\n' "$words" | while IFS= read -r l; do
    case $l in ("$p:"*) [ -z "$t" ] || case $l in (*"$t"*) ;; (*) continue ;; esac; printf '%s\n' "$l" ;; esac
  done)
  if [ -z "$hit" ]; then wunmatched+=("$a"); continue; fi
  wallowed=$((wallowed + $(printf '%s\n' "$hit" | wc -l)))
  words=$(printf '%s\n' "$words" | g -v -xF -- "$hit")
done
if [ "${#wunmatched[@]}" -gt 0 ]; then
  printf 'check-comments: WORD_ALLOW entry matches no line: %s\n' "${wunmatched[@]}" >&2
  fail=1
fi
if [ -n "$words" ]; then
  n=$(printf '%s\n' "$words" | wc -l | tr -d ' ')
  nf=$(printf '%s\n' "$words" | cut -d: -f1 | LC_ALL=C sort -u | wc -l | tr -d ' ')
  sed -n 1,40p <<<"$words" >&2
  echo "check-comments: $n line(s) in $nf file(s) of the tree carry a retired word ($RETIRED_WORDS, any case) — the feature is deleted; describe what runs now" >&2
  fail=1
fi
[ "$fail" = 0 ] || exit 1
echo "check-comments: ok (${#files[@]} files, $(printf '%s\n' "$comments" | wc -l | tr -d ' ') comment lines; tools/ cites no doc by line number, $allowed allowed by CITE_ALLOW; no retired word, $wallowed line(s) allowed by WORD_ALLOW)"
