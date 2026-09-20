#!/usr/bin/env bash
# 박스에 남은 트랙 디렉터리(~/repo/bloomery-<track>)를 로컬 워크트리와 대조한다(맥에서 실행).
#   tools/box-tracks.sh            # 목록만: 크기와 상태(live = 로컬 워크트리 있음, stale = 없음)
#   tools/box-tracks.sh --remove   # stale만 지운다. 메인 디렉터리와 live 트랙은 절대 건드리지 않는다.
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
  else
    echo "stale  $size  $name"
    stale+=("$name")
  fi
done < <(ssh "$HOST" "cd ~/repo && du -sh ${MAIN} ${MAIN}-* 2>/dev/null")
[ "${1:-}" = "--remove" ] || { echo "${#stale[@]} stale — rerun with --remove to delete them"; exit 0; }
for name in "${stale[@]}"; do
  case "$name" in "$MAIN"-?*) ;; *) echo "refusing odd name: $name" >&2; exit 1 ;; esac
  # 그 디렉터리 아래 실행 파일을 문 프로세스가 있으면 지우지 않는다(돌고 있는 트랙일 수 있다).
  if ssh "$HOST" "pgrep -f \"\$HOME/repo/$name/target\" >/dev/null"; then
    echo "skip   $name — a process still runs from its target/" >&2
    continue
  fi
  ssh "$HOST" "rm -rf ~/repo/$name" && echo "removed $name"
done
