#!/usr/bin/env bash
# mulle — 박스(워크스테이션)에서 빌드·실행하는 러너.
# 소스 트리를 박스의 ~/repo/mulle로 rsync한 뒤 인자를 그 디렉터리에서 실행한다.
# 환경(nightly, LLVM 21 타르볼, CUDA 13.0, 3090 핀)은 박스의 ~/mulle-env.sh가 소유한다.
# 데이터·참조 바이너리 디렉터리는 MULLE_DATA 하나가 소유한다(기본 /root/mulle-data). 병렬 트랙은 이 값과 REMOTE만 바꾼다.
#   tools/box.sh cargo oxide doctor
#   tools/box.sh cargo oxide run q3k_gemv --arch sm_86
set -euo pipefail
HOST=${MULLE_BOX:-ws}
REMOTE=${MULLE_REMOTE:-'~/repo/mulle-m8-glm'}
HERE=$(cd "$(dirname "$0")/.." && pwd)
ssh "$HOST" "mkdir -p $REMOTE"
rsync -az --delete --exclude target/ --exclude .git/ "$HERE"/ "$HOST:$REMOTE/"
DATA=${MULLE_DATA:-/root/mulle-data-m8-glm}
ssh "$HOST" "source ~/mulle-env.sh && export MULLE_DATA=$DATA && cd $REMOTE && $*"
