#!/usr/bin/env bash
# 박스에 남은 트랙 디렉터리(~/repo/bloomery-<track>)를 로컬 워크트리와 대조한다(맥에서 실행).
#   tools/box-tracks.sh            # 목록만: 크기와 상태(live = 로컬 워크트리 있음, stale = 없음,
#                                  # aux = 살아 있는 트랙 이름에 `-접미사`를 붙인 그 트랙의 보조 트리)
#   tools/box-tracks.sh --remove   # stale만 지운다. 메인 디렉터리·live·aux는 절대 건드리지 않고, 그 안에서
#                                  # 프로세스가 도는(실행 파일이 target/ 아래이거나 cwd가 그 안인) stale도 건너뛴다.
# 막는 실패: 워크트리는 지웠는데 원격 디렉터리(target/ 포함 수백 MB)가 남아 쌓이는 것.
# 2026-09-20 하루에 14개 7.6 GB가 남았다 — 트랙 체크리스트의 마지막 단계가 이 스크립트다.
set -euo pipefail
HOST=${BLOOMERY_BOX:-ws}
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$(basename "$(git -C "$HERE" worktree list --porcelain | awk '/^worktree /{print $2; exit}')")
live=$(git -C "$HERE" worktree list --porcelain | awk '/^worktree /{print $2}' | xargs -n1 basename)
stale=()
while read -r size dir; do
  [ -n "$dir" ] || continue
  name=$(basename "$dir")
  if [ "$name" = "$MAIN" ] || printf '%s\n' "$live" | grep -qx "$name"; then
    echo "live   $size  $name"
    continue
  fi
  # 살아 있는 트랙 이름에 `-접미사`를 붙인 디렉터리는 그 트랙의 보조 트리다(git archive로 뜬
  # 기준 트리 같은 것 — 로컬 워크트리가 없다). 트랙이 끝나 워크트리가 사라지면 함께 stale이 된다.
  owner=
  while read -r l; do
    [ -n "$l" ] && [ "$l" != "$MAIN" ] || continue
    case "$name" in "$l"-?*) owner=$l ;; esac
  done <<< "$live"
  if [ -n "$owner" ]; then
    echo "aux    $size  $name (live track $owner)"
  else
    echo "stale  $size  $name"
    stale+=("$name")
  fi
done < <(ssh "$HOST" "cd ~/repo && du -sh ${MAIN} ${MAIN}-* 2>/dev/null")
[ "${1:-}" = "--remove" ] || { echo "${#stale[@]} stale — rerun with --remove to delete them"; exit 0; }
for name in ${stale[@]+"${stale[@]}"}; do  # bash 3.2 + set -u: an empty array is unbound
  case "$name" in "$MAIN"-?*) ;; *) echo "refusing odd name: $name" >&2; exit 1 ;; esac
  # 그 디렉터리에서 도는 프로세스가 있으면 지우지 않는다(돌고 있는 트랙일 수 있다). 판정은 box-gc.sh의
  # --check다: 실행 파일이 target/ 아래(테스트 바이너리)이거나 cwd가 그 안(빌드 중인 cargo·rustc — 실행
  # 파일은 rustup 쪽이라 exe만 보면 안 걸린다). cmdline 매칭은 자기 셸과 ssh 자식을 같이 고른다.
  # stale 디렉터리에는 그 스크립트가 없을 수 있으므로 stdin으로 넘긴다.
  rc=0
  ssh "$HOST" "bash -s -- --check \"\$HOME/repo/$name\"" < "$HERE/tools/box-gc.sh" > /dev/null || rc=$?
  if [ "$rc" = 10 ]; then
    echo "skip   $name — a process still runs there (exe under its target/, or cwd inside)" >&2
    continue
  elif [ "$rc" != 0 ]; then
    echo "skip   $name — the process scan failed (rc $rc)" >&2
    continue
  fi
  ssh "$HOST" "rm -rf ~/repo/$name" && echo "removed $name"
done
