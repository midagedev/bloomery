#!/usr/bin/env bash
# 박스에 남은 트랙 디렉터리(~/repo/bloomery-<track>)를 로컬 워크트리와 대조한다(맥에서 실행).
#   tools/box-tracks.sh                  # 목록만: 크기와 상태(live = 로컬 워크트리 있음, stale = 없음,
#                                        # aux = 살아 있는 트랙 이름에 `-접미사`를 붙인 그 트랙의 보조 트리)
#   tools/box-tracks.sh --remove NAME…   # 이름을 준 디렉터리만 지운다. 지우는 순간 다시 분류해서 그때도
#                                        # stale인 것만 지우고, 메인 디렉터리·live·aux는 절대 건드리지 않고, 그 안에서
#                                        # 프로세스가 도는(실행 파일이 target/ 아래이거나 cwd가 그 안인) stale도 건너뛴다.
#                                        # 이름을 안 준 새 stale은 이름을 찍고 건너뛴다. 이름 없는 --remove는 거절(64).
#   tools/box-tracks.sh --self-test      # 분류와 삭제 선택을 고정 입력으로 검사한다(ssh 없음; check-recipes가 돌린다).
# 막는 실패 둘: 워크트리는 지웠는데 원격 디렉터리(target/ 포함 수백 MB)가 남아 쌓이는 것, 그리고 사람이 읽은
# 목록과 실제로 지우는 집합이 다른 것 — 목록을 본 뒤 --remove까지 사이에 다른 세션이 워크트리를 지우면, 이름 없이
# "stale 전부"를 지우던 옛 형태는 그 세션이 아직 쓰려던 박스 디렉터리까지 지웠다.
set -euo pipefail

# stdin: `du -sh` 줄("<size>\t<path>"). $1 = 메인 디렉터리 이름, $2 = 로컬 워크트리 이름들(줄마다 하나).
# 출력: "<state> <size> <name> [<owner>]" — state는 live·aux·stale.
classify() {
  local main=$1 live=$2 size dir name owner l
  while read -r size dir; do
    [ -n "$dir" ] || continue
    name=$(basename "$dir")
    if [ "$name" = "$main" ] || printf '%s\n' "$live" | grep -qx "$name"; then
      echo "live $size $name"
      continue
    fi
    # 살아 있는 트랙 이름에 `-접미사`를 붙인 디렉터리는 그 트랙의 보조 트리다(git archive로 뜬
    # 기준 트리 같은 것 — 로컬 워크트리가 없다). 트랙이 끝나 워크트리가 사라지면 함께 stale이 된다.
    owner=
    while read -r l; do
      [ -n "$l" ] && [ "$l" != "$main" ] || continue
      case "$name" in "$l"-?*) owner=$l ;; esac
    done <<< "$live"
    if [ -n "$owner" ]; then echo "aux $size $name $owner"; else echo "stale $size $name"; fi
  done
}

# $1 = 메인 이름, $2 = classify 출력, 나머지 = 지우라고 준 이름(`bloomery-` 접두사는 생략 가능).
# 출력: "rm <name>" 또는 "skip <name> <이유>". 준 이름 중 하나라도 지울 수 없으면 rc 1.
select_removals() {
  local main=$1 table=$2 rc=0 want name state seen
  shift 2
  seen=
  for want in "$@"; do
    case "$want" in "$main"-?*) name=$want ;; *) name="$main-$want" ;; esac
    state=$(awk -v n="$name" '$3 == n {print $1}' <<< "$table")
    case "$state" in
      stale) echo "rm $name" ;;
      '') echo "skip $name not-on-the-box"; rc=1 ;;
      *) echo "skip $name $state-now"; rc=1 ;;
    esac
    seen="$seen $name "
  done
  while read -r state _ name _; do
    [ "$state" = stale ] || continue
    case "$seen" in *" $name "*) ;; *) echo "skip $name stale-but-not-named" ;; esac
  done <<< "$table"
  return $rc
}

self_test() {
  local fails=0 table out rc
  local du=$'1.2G\t./bloomery\n300M\t./bloomery-a\n10M\t./bloomery-a-base\n400M\t./bloomery-b\n500M\t./bloomery-c\n20M\t./bloomery-b-base'
  table=$(classify bloomery $'bloomery\nbloomery-a' <<< "$du")
  check() { if [ "$2" != "$3" ]; then echo "FAIL $1: got [$2] want [$3]" >&2; fails=$((fails + 1)); else echo "ok $1"; fi; }
  check classify "$(awk '{print $1, $3, $4}' <<< "$table" | tr '\n' ';')" \
    "live bloomery ;live bloomery-a ;aux bloomery-a-base bloomery-a;stale bloomery-b ;stale bloomery-c ;stale bloomery-b-base ;"
  # 이름을 준 stale 하나만 지우고, 나머지 새 stale은 이름을 찍고 건너뛴다(rc 0).
  rc=0; out=$(select_removals bloomery "$table" b) || rc=$?
  check named-only "$(tr '\n' ';' <<< "$out") rc=$rc" \
    "rm bloomery-b;skip bloomery-c stale-but-not-named;skip bloomery-b-base stale-but-not-named; rc=0"
  # 목록을 본 뒤 그 이름이 live·aux가 됐거나 사라졌으면 지우지 않고 rc 1.
  rc=0; out=$(select_removals bloomery "$table" bloomery-a a-base gone bloomery) || rc=$?
  check not-stale-now "$(grep -c '^rm' <<< "$out" || true) $(grep '^skip bloomery-a ' <<< "$out") $(grep '^skip bloomery-gone' <<< "$out") rc=$rc" \
    "0 skip bloomery-a live-now skip bloomery-gone not-on-the-box rc=1"
  # 메인 이름을 주면 접두사 규칙으로 "bloomery-bloomery"가 되어 박스에 없음으로 거절된다.
  check main-refused "$(grep -c '^rm bloomery$' <<< "$out" || true)" "0"
  # 이름 없는 --remove는 거절.
  rc=0; bash "$0" --remove > /dev/null 2>&1 || rc=$?
  check bare-remove-refused "$rc" 64
  [ "$fails" = 0 ] && echo "box-tracks: self-test ok" || { echo "box-tracks: self-test $fails failed" >&2; return 1; }
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --remove)
    shift
    if [ $# = 0 ]; then
      echo "box-tracks: --remove takes the names to delete (as the listing prints them); a bare --remove is refused" >&2
      exit 64
    fi ;;
  '') ;;
  *) echo "box-tracks: unknown argument $1 (none, --remove NAME…, --self-test)" >&2; exit 64 ;;
esac

HOST=${BLOOMERY_BOX:-ws}
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$(basename "$(git -C "$HERE" worktree list --porcelain | awk '/^worktree /{print $2; exit}')")
live=$(git -C "$HERE" worktree list --porcelain | awk '/^worktree /{print $2}' | xargs -n1 basename)
table=$(ssh "$HOST" "cd ~/repo && du -sh ${MAIN} ${MAIN}-* 2>/dev/null" | classify "$MAIN" "$live")
awk '{ printf "%-6s %6s  %s%s\n", $1, $2, $3, ($4 != "" ? " (live track " $4 ")" : "") }' <<< "$table"
nstale=$(awk '$1 == "stale"' <<< "$table" | wc -l | tr -d ' ')
if [ $# = 0 ]; then
  echo "$nstale stale — delete by name: tools/box-tracks.sh --remove NAME…"
  exit 0
fi

rc=0
plan=$(select_removals "$MAIN" "$table" "$@") || rc=$?
while read -r verb name why; do
  [ -n "$verb" ] || continue
  if [ "$verb" = skip ]; then echo "skip   $name — $why" >&2; continue; fi
  case "$name" in "$MAIN"-?*) ;; *) echo "refusing odd name: $name" >&2; exit 1 ;; esac
  # 그 디렉터리에서 도는 프로세스가 있으면 지우지 않는다(돌고 있는 트랙일 수 있다). 판정은 box-gc.sh의
  # --check다: 실행 파일이 target/ 아래(테스트 바이너리)이거나 cwd가 그 안(빌드 중인 cargo·rustc — 실행
  # 파일은 rustup 쪽이라 exe만 보면 안 걸린다). cmdline 매칭은 자기 셸과 ssh 자식을 같이 고른다.
  # stale 디렉터리에는 그 스크립트가 없을 수 있으므로 stdin으로 넘긴다.
  crc=0
  ssh "$HOST" "bash -s -- --check \"\$HOME/repo/$name\"" < "$HERE/tools/box-gc.sh" > /dev/null || crc=$?
  if [ "$crc" = 10 ]; then
    echo "skip   $name — a process still runs there (exe under its target/, or cwd inside)" >&2; rc=1; continue
  elif [ "$crc" != 0 ]; then
    echo "skip   $name — the process scan failed (rc $crc)" >&2; rc=1; continue
  fi
  # -n: without it ssh reads the loop's stdin (the plan) and every later name is lost.
  ssh -n "$HOST" "rm -rf ~/repo/$name" && echo "removed $name"
done <<< "$plan"
exit $rc
