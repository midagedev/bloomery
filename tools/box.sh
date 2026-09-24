#!/usr/bin/env bash
# bloomery — 박스(워크스테이션)에서 빌드·실행하는 러너.
# 소스 트리를 박스의 ~/repo/bloomery로 rsync한 뒤 인자를 그 디렉터리에서 실행한다.
# 환경(nightly, LLVM 21 타르볼, CUDA 13.3, 3090 핀)은 박스의 ~/bloomery-env.sh가 소유한다.
# 데이터·참조 바이너리 디렉터리는 BLOOMERY_DATA 하나가 소유한다. 병렬 트랙은 이 값과 REMOTE만 바꾼다.
#   tools/box.sh cargo oxide doctor
#   tools/box.sh cargo oxide run q3k_gemv --arch sm_86
set -euo pipefail
HOST=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$(cd "$(dirname "$0")/.." && pwd)")"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
ssh "$HOST" "mkdir -p $REMOTE"
# 시각은 싣지 않는다(-t 없음) — 바뀐 파일은 내용 체크섬(-c)으로 고르고, 박스에 닿은 파일의 mtime은 박스 시계의 "지금"이 된다.
# 맥의 mtime을 그대로 실으면 cargo가 낡은 바이너리를 내준다: 박스 시계가 맥보다 앞서 있어(실측 4.1초) 복원 직후의 touch조차
# 직전 빌드 산출물보다 과거로 찍힌다(변이 바이너리가 두 번 그대로 돌았다).
# 원격 루트의 *.ptx·*.ll은 cargo oxide가 빌드 중에 쓰는 산출물이다(`bloomery_gpu_deepseek41.ptx`, `….linked.opt.ll`) — 같은
# 원격 디렉터리에서 빌드가 도는 사이 다른 box.sh 호출의 --delete가 그것을 지우면 빌드가 rc 101로 죽는다(dspark-q3k와
# ds41splitk 라운드에서 한 번씩). 맥 트리에는 없으니 삭제 대상에서 뺀다.
rsync -rlpgoDcz --delete --exclude target/ --exclude .git/ --exclude '/*.ptx' --exclude '/*.ll' "$HERE"/ "$HOST:$REMOTE/"
# 카드 선택. 기본은 env 파일의 3090 핀 그대로. BLOOMERY_CARD=a6000|both는 박스에서 이름으로 UUID를 찾아
# CUDA_VISIBLE_DEVICES를 덮어쓴다(both = 3090 먼저 → 디바이스 0이 3090). 두 카드 다 우리 것이다(야간 학습은
# 2026-09-21에 끝났고 llm.service는 꺼져 있다). 그래도 그 카드에 이미 컴퓨트 프로세스가 있으면 — 우리 다른
# 라운드일 것이다 — 겹쳐 올리지 않고 rc 75로 끝난다. llm.service 검사는 누가 다시 켰을 때의 안전장치다.
CARD=${BLOOMERY_CARD:-3090}
case "$CARD" in
  3090) PICK=":" ;;
  a6000|both) PICK='
    A=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "A6000" | cut -d, -f1)
    T=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "3090" | cut -d, -f1)
    # a6000 alone needs only its own UUID — the 3090 has fallen off the bus twice on 2026-09-22 and must not
    # take the healthy card down with it. both needs both.
    [ -n "$A" ] || { echo "box.sh: A6000 lookup failed" >&2; exit 75; }
    [ "'"$CARD"'" != both ] || [ -n "$T" ] || { echo "box.sh: 3090 lookup failed (both)" >&2; exit 75; }
    if [ "$(systemctl is-active llm.service)" = active ]; then echo "box.sh: llm.service holds the A6000" >&2; exit 75; fi
    if [ -n "$(nvidia-smi -i "$A" --query-compute-apps=pid --format=csv,noheader)" ]; then
      echo "box.sh: the A6000 has compute processes (serving or training) — not taking it" >&2; exit 75; fi
    '"$( [ "$CARD" = both ] && echo 'export CUDA_VISIBLE_DEVICES="$T,$A"' || echo 'export CUDA_VISIBLE_DEVICES="$A"' )" ;;
  *) echo "box.sh: BLOOMERY_CARD must be 3090, a6000 or both" >&2; exit 64 ;;
esac
# The model the gates and GPU binaries open (BLOOMERY_REF_MODEL) is a property of the tool profile:
# every command runs with it exported from tools/ref/ref-paths.sh, and an unknown profile stops the
# command with that file's exit 64. The profile is picked on this side — BLOOMERY_MODEL here, or
# ref-paths.sh's default — and its name goes along as BLOOMERY_REF_MODEL_PROFILE, which the scripts
# inside take as their profile; one that picks another is refused (ref-paths.sh, exit 64), since
# the export would not follow it. BLOOMERY_MODEL itself is not exported: the model crate's test
# harnesses read that name as a model path. A caller's own BLOOMERY_REF_MODEL wins: one set on this
# side is carried over, one set inside the command overrides the export.
FWD=
if [ -n "${BLOOMERY_REF_MODEL:-}" ]; then
  FWD="export BLOOMERY_REF_MODEL=$(printf %q "$BLOOMERY_REF_MODEL") && "
fi
MODEL_PICK=
if [ -n "${BLOOMERY_MODEL:-}" ]; then
  MODEL_PICK="BLOOMERY_MODEL=$(printf %q "$BLOOMERY_MODEL") && "
fi
PROFILE="__p=\$(${MODEL_PICK}. tools/ref/ref-paths.sh && printf %s \"\$BLOOMERY_MODEL\") && export BLOOMERY_REF_MODEL_PROFILE=\"\$__p\" && __m=\$(. tools/ref/ref-paths.sh && printf %s \"\$MODEL\") && export BLOOMERY_REF_MODEL=\"\$__m\" && unset __p __m"
# The V4.1 file (BLOOMERY_V41_MODEL, and its directory as BLOOMERY_V41_DIR) is exported into every command
# whatever profile it picked: the deepseek41 profile owns the choice (its V41_MODEL), and the crates and
# scripts that open V4.1 under another profile (the tokenizer, engram, qdot and placement tests, the
# tokenizer and engram runners) read the export. A caller's own BLOOMERY_V41_MODEL is carried over and wins.
V41=
if [ -n "${BLOOMERY_V41_MODEL:-}" ]; then
  V41="export BLOOMERY_V41_MODEL=$(printf %q "$BLOOMERY_V41_MODEL") && "
fi
V41="${V41}__v=\$(. tools/ref/models/deepseek41.sh && printf %s \"\$V41_MODEL\") && export BLOOMERY_V41_MODEL=\"\$__v\" BLOOMERY_V41_DIR=\"\${__v%/*}\" && unset __v"
# The data directory (BLOOMERY_DATA) is exported into every command the same way: its default is
# ref-paths.sh's, read on the box from the synced tree, so no copy of it lives here. A caller's own
# BLOOMERY_DATA is carried over and wins.
DATA=
if [ -n "${BLOOMERY_DATA:-}" ]; then
  DATA="export BLOOMERY_DATA=$(printf %q "$BLOOMERY_DATA") && "
fi
DATA="${DATA}__d=\$(. tools/ref/ref-paths.sh && printf %s \"\$BLOOMERY_DATA\") && export BLOOMERY_DATA=\"\$__d\" && unset __d"
# BLOOMERY_BOX_ENV="NAME=value NAME2=value2" exports those variables into the command, so a lever arm
# reaches the binary through an unchanged recipe (tools/gpu-ab.py's env arms). Entries are split on
# spaces; each value is quoted for the remote shell.
ENVS=
read -r -a box_env <<< "${BLOOMERY_BOX_ENV:-}"
for kv in ${box_env[@]+"${box_env[@]}"}; do
  name=${kv%%=*}
  case "$kv" in
    *=*) ;;
    *) echo "box.sh: BLOOMERY_BOX_ENV entries are NAME=value, got '$kv'" >&2; exit 64 ;;
  esac
  case "$name" in
    '' | [0-9]* | *[!A-Za-z0-9_]*) echo "box.sh: '$name' in BLOOMERY_BOX_ENV is not a variable name" >&2; exit 64 ;;
  esac
  ENVS="${ENVS}export $name=$(printf %q "${kv#*=}") && "
done
# serve의 build.rs가 `/props`의 version에 새기는 커밋. 박스 사본에는 .git이 없어서(rsync가 뺀다) 여기서 넘긴다.
# 이 트리에 커밋에 없는 변경이 있으면 `-dirty`를 붙인다 — 그 바이너리는 그 커밋의 것이 아니다.
COMMIT=$(git -C "$HERE" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
[ -z "$(git -C "$HERE" status --porcelain 2>/dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
ssh "$HOST" "source ~/bloomery-env.sh && { $PICK
} && cd $REMOTE && $V41 && $FWD$PROFILE && $DATA && export BLOOMERY_GIT_COMMIT=$COMMIT && $ENVS$*"
