#!/usr/bin/env bash
# bloomery — 박스(워크스테이션)에서 빌드·실행하는 러너.
# 소스 트리를 박스의 ~/repo/bloomery로 rsync한 뒤 인자를 그 디렉터리에서 실행한다.
# 환경(nightly, LLVM 21 타르볼, CUDA 13.3, 3090 핀)은 박스의 ~/bloomery-env.sh가 소유한다.
# 데이터·참조 바이너리 디렉터리는 BLOOMERY_DATA 하나가 소유한다. 병렬 트랙은 이 값과 REMOTE만 바꾼다.
#   tools/box.sh cargo oxide doctor
#   tools/box.sh cargo oxide run q3k_gemv --arch sm_86
#
# The guard. Before the command, on the box and in the same ssh, lease_guard (tools/ref/lease-probe.sh)
# reads the timing lease (/root/bloomery-cpu.lock, through lease_free: a shared-lock probe) and every
# hold (/root/bloomery-<owner>-hold, up while the file exists). While either is up the command does
# not start: a build or a gate beside a sitting puts [cpu-busy] on the sitting's rows and voids them.
# It waits, naming what is up — the lease's holders, each hold's owner and age — at once and once a
# minute, polls every 30 s, and after a wait starts only on two quiet polls in a row (a sitting of
# several runs leaves the lease free for seconds between them). Still busy at the bound: exit 75,
# contention, naming what is up. A lease that cannot be tested: 70. The guard holds nothing while
# the command runs, so a runner in the command takes the lease after it as before.
#   BLOOMERY_BOX_WAIT=<s>     the bound, default 1800: the wait lease_take and gpu-gate.sh give up
#                             after. 0 does not wait: 75 at once when busy (gate-batch.sh's checks)
#   BLOOMERY_HOLD_OWNER=<o>   this call belongs to the owner of /root/bloomery-<o>-hold and passes that
#                             hold — a sitting script puts its hold up and runs its runners through
#                             box.sh; the lease and every other hold still stop it, and with its own
#                             hold up and another that went up first it gives way at once (75)
#   BLOOMERY_TIER=real|fixture  the tier the command runs in (unset: real, and every byte of the remote command as before). It
#                             names the tier the model file comes from: tools/ref/ref-paths.sh resolves BLOOMERY_REF_MODEL
#                             to the family's fixture under `fixture`, exports the fixture's card budget as BLOOMERY_CARD_BUDGET
#                             and the file as BLOOMERY_FIXTURE_MODEL (set only when the tier resolved one; tools/gpu-gate.sh
#                             skips its V4.1 load lock only for that file), and refuses a family with none (66) — it never runs
#                             one on its real file. The tier comes from the environment here or a BLOOMERY_TIER entry of
#                             BLOOMERY_BOX_ENV, which is read here, before the profile: an entry the profile had not seen would
#                             reach the binary and not the file. Both set to different tiers is 64.
#   BLOOMERY_BOX_READONLY=1   a read — ps, cat, tail, ls, nvidia-smi, a status probe — runs without the
#                             guard: the way to read a sitting's log, the owner's own included, while it
#                             runs. Refused (64) when the command names cargo, just, make, cmake, ninja
#                             or target/, so the opt-in cannot carry a build or one of our binaries. It
#                             syncs nothing and creates nothing: the command runs in the remote directory
#                             as it is, and a remote directory that is not there is refused (66) — a
#                             sync beside a sitting would put this tree's uncommitted edits under the
#                             sitting's running scripts. Its ssh is the only one that reads stdin, so a
#                             script can go over it (`just box-gc`: 'bash -s -- …' < tools/box-gc.sh)
set -euo pipefail
HOST=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$(cd "$(dirname "$0")/.." && pwd)")"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BOX_WAIT=${BLOOMERY_BOX_WAIT:-1800}
case $BOX_WAIT in
  '' | *[!0-9]*) echo "box.sh: BLOOMERY_BOX_WAIT is whole seconds (0: do not wait), got '$BOX_WAIT'" >&2; exit 64 ;;
esac
HOLD_OWNER=${BLOOMERY_HOLD_OWNER:-}
case $HOLD_OWNER in
  *[!A-Za-z0-9_]*) echo "box.sh: BLOOMERY_HOLD_OWNER is the <owner> of /root/bloomery-<owner>-hold (letters, digits, _), got '$HOLD_OWNER'" >&2; exit 64 ;;
esac
# The tier (the header), named before anything is synced: a refusal touches nothing on the box.
TIER_MAC=${BLOOMERY_TIER:-} TIER_BOX=
read -r -a tier_env <<< "${BLOOMERY_BOX_ENV:-}"
for kv in ${tier_env[@]+"${tier_env[@]}"}; do
  [ "${kv%%=*}" != BLOOMERY_TIER ] || TIER_BOX=${kv#*=}
done
for t in "$TIER_MAC" "$TIER_BOX"; do
  case "$t" in
    '' | real | fixture) ;;
    *) echo "box.sh: BLOOMERY_TIER is real or fixture (unset: real), got '$t'" >&2; exit 64 ;;
  esac
done
if [ -n "$TIER_MAC" ] && [ -n "$TIER_BOX" ] && [ "$TIER_MAC" != "$TIER_BOX" ]; then
  echo "box.sh: BLOOMERY_TIER is '$TIER_MAC' in the environment and '$TIER_BOX' in BLOOMERY_BOX_ENV: name one" >&2
  exit 64
fi
TIER=${TIER_BOX:-${TIER_MAC:-real}}
# Exported ahead of the profile below, so ref-paths.sh sees it; nothing is added for a command that names none.
TIERX=
[ -z "$TIER_MAC$TIER_BOX" ] || TIERX="export BLOOMERY_TIER=$TIER && "
READONLY=0
case ${BLOOMERY_BOX_READONLY:-0} in
  0) GUARD="( cd $REMOTE && . tools/ref/lease-probe.sh && lease_guard $BOX_WAIT $HOLD_OWNER ) && " ;;
  1)
    READONLY=1
    GUARD=
    ro_build='(^|[^A-Za-z0-9_])(cargo|just|make|cmake|ninja)([^A-Za-z0-9_]|$)'
    if [[ $* =~ $ro_build ]]; then
      echo "box.sh: BLOOMERY_BOX_READONLY=1 is for reads, and this command names '${BASH_REMATCH[2]}': run it without the opt-in, behind the guard" >&2
      exit 64
    fi
    case $* in
      *target/*) echo "box.sh: BLOOMERY_BOX_READONLY=1 is for reads, and this command names 'target/' (one of our binaries): run it without the opt-in, behind the guard" >&2; exit 64 ;;
    esac
    ;;
  *) echo "box.sh: BLOOMERY_BOX_READONLY is 1 (a read), or 0 or unset (behind the guard), got '$BLOOMERY_BOX_READONLY'" >&2; exit 64 ;;
esac
if [ "$READONLY" = 1 ]; then
  # A read runs in the remote directory as it is. -n: this lookup leaves stdin to the command's ssh.
  rc=0
  ssh -n "$HOST" "test -d $REMOTE" || rc=$?
  case $rc in
    0) ;;
    1) echo "box.sh: BLOOMERY_BOX_READONLY=1 runs the command in $REMOTE as it is and syncs nothing, and $HOST has no $REMOTE: nothing was synced there yet" >&2; exit 66 ;;
    *) echo "box.sh: ssh $HOST failed looking for $REMOTE (rc $rc)" >&2; exit "$rc" ;;
  esac
else
  ssh -n "$HOST" "mkdir -p $REMOTE"
  # 시각은 싣지 않는다(-t 없음) — 바뀐 파일은 내용 체크섬(-c)으로 고르고, 박스에 닿은 파일의 mtime은 박스 시계의 "지금"이 된다.
  # 맥의 mtime을 그대로 실으면 cargo가 낡은 바이너리를 내준다: 박스 시계가 맥보다 앞서 있어(실측 4.1초) 복원 직후의 touch조차
  # 직전 빌드 산출물보다 과거로 찍힌다(변이 바이너리가 두 번 그대로 돌았다).
  # 원격 루트의 *.ptx·*.ll은 cargo oxide가 빌드 중에 쓰는 산출물이다(`bloomery_gpu_deepseek41.ptx`, `….linked.opt.ll`) — 같은
  # 원격 디렉터리에서 빌드가 도는 사이 다른 box.sh 호출의 --delete가 그것을 지우면 빌드가 rc 101로 죽는다(dspark-q3k와
  # ds41splitk 라운드에서 한 번씩). `.oxide-artifacts/`(`embed.o` 등)도 같은 부류다(q3fix 라운드에서 한 번). 맥 트리에는 없으니
  # 삭제 대상에서 뺀다.
  rsync -rlpgoDcz --delete --exclude target/ --exclude .git/ --exclude '/*.ptx' --exclude '/*.ll' --exclude '/.oxide-artifacts/' "$HERE"/ "$HOST:$REMOTE/"
fi
# 카드 선택. 기본은 env 파일의 3090 핀 그대로. BLOOMERY_CARD=a6000|both는 박스에서 이름으로 UUID를 찾아
# CUDA_VISIBLE_DEVICES를 덮어쓴다(both = 3090 먼저 → 디바이스 0이 3090). 두 카드 다 우리 것이다(야간 학습은
# 2026-09-21에 끝났고 llm.service는 꺼져 있다). A compute process on the A6000 with its gate lock and the timing
# lease both free is not one of ours: the command ends with rc 75 (below). llm.service 검사는 누가 다시 켰을 때의
# 안전장치다.
# 고른 값은 박스 쪽에 BLOOMERY_BOX_CARD로 넘어가고, tools/gpu-gate.sh가 그 카드의 게이트 락을 잡는다(both = 둘 다).
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
    # A compute process on the A6000 while its gate lock or the timing lease is held is one of our runs: the
    # command goes on, and tools/gpu-gate.sh waits for the card lock inside its bound. One with both free is
    # a surprise (serving or training) and ends the command with 75. Shared probes: they never hold a lock.
    if [ -n "$(nvidia-smi -i "$A" --query-compute-apps=pid --format=csv,noheader)" ] &&
       flock -s -n /root/bloomery-gate-a6000.lock true && flock -s -n /root/bloomery-cpu.lock true; then
      echo "box.sh: the A6000 has compute processes and neither its gate lock nor the timing lease is held (serving or training?) — not taking it" >&2; exit 75; fi
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
# The fixture tier also takes the fixture file's card budget and names the file, in one more read of ref-paths.sh each; a family
# whose real file stands (FIXTURE_FILE empty) exports neither, and a file that is not a whole fixture stops the command (65).
if [ "$TIER" = fixture ]; then
  PROFILE="$PROFILE && __f=\$(. tools/ref/ref-paths.sh && printf %s \"\$FIXTURE_FILE\") && __b=\$(. tools/ref/ref-paths.sh && fixture_budget) && { [ -z \"\$__f\" ] || export BLOOMERY_FIXTURE_MODEL=\"\$__f\" BLOOMERY_CARD_BUDGET=\"\$__b\"; } && unset __f __b"
fi
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
# spaces and quote nothing; each value is quoted for the remote shell. So a value with a space cannot
# cross: an entry that holds a quote, or one that is not an upper-case NAME=value (the second half of a
# spaced value), is refused by name — such a value goes in the command itself, or in an arm word (a
# server arm's +t<N>, +nopo<0|1>, +k<K>: tools/ref/lcpp-warm.sh).
ENVS=
read -r -a box_env <<< "${BLOOMERY_BOX_ENV:-}"
prev=
spaced="BLOOMERY_BOX_ENV splits on spaces and quotes nothing, so a value with a space does not cross it; put it in the command, or in an arm word (a server arm's +t<N>, +nopo<0|1>, +k<K>)"
for kv in ${box_env[@]+"${box_env[@]}"}; do
  name=${kv%%=*} after=
  [ -z "$prev" ] || after=" (after '$prev': the rest of a value with a space? $spaced)"
  case "$kv" in
    *[\"\']*) echo "box.sh: BLOOMERY_BOX_ENV entry '$kv' holds a quote: $spaced" >&2; exit 64 ;;
    *=*) ;;
    *) echo "box.sh: BLOOMERY_BOX_ENV entries are NAME=value, got '$kv'$after" >&2; exit 64 ;;
  esac
  case "$name" in
    '' | [0-9]* | *[![:upper:][:digit:]_]*) echo "box.sh: '$name' in BLOOMERY_BOX_ENV is not an upper-case variable name$after" >&2; exit 64 ;;
  esac
  ENVS="${ENVS}export $name=$(printf %q "${kv#*=}") && "
  prev=$kv
done
# 트랙 원격 디렉터리(BLOOMERY_REMOTE)의 빌드는 CARGO_BUILD_JOBS 기본 12: 파동의 4중 빌드가 32코어를 스래싱하지 않게.
# BLOOMERY_BOX_ENV로 주면 그 값이고, 메인 트리(BLOOMERY_REMOTE 없음)는 그대로다.
if [ -n "${BLOOMERY_REMOTE:-}" ] && [[ " ${box_env[*]+${box_env[*]}} " != *" CARGO_BUILD_JOBS="* ]]; then
  ENVS="${ENVS}export CARGO_BUILD_JOBS=12 && "
fi
# cuda-oxide 백엔드(librustc_codegen_cuda.so)는 핀 rev마다 제 디렉터리에 둔다: ~/.cargo/cuda-oxide-bloomery/<rev>/.
# cargo-oxide의 기본 캐시(~/.cargo/cuda-oxide/)는 소스가 새로워 보이면 그 자리에서 다시 빌드하고, 그동안 다른 트랙의
# rustc가 그 .so를 mmap한 채 SIGBUS를 맞는다(nvlabs-ledger 19). rev로 고정한 경로의 파일은 한 번 놓이면 바뀌지 않는다.
# rev는 Cargo.toml [patch."https://github.com/NVIDIA/cuda-rust.git"]의 포크 rev이고, 그 절이 없으면
# [workspace.dependencies]의 rev다. cargo oxide를 부르는 명령은 그 파일이 없거나 옆의 source-rev.txt가 그 rev가 아니면 이름을 대고 멈춘다(rc 70) — CUDA_OXIDE_BACKEND로
# 고정하면 cargo-oxide는 백엔드와 의존성의 커밋을 대조하지 않으므로 그 대조는 여기가 한다. 새 rev의 백엔드는 그 rev의 체크아웃(~/.cargo/git/checkouts/
# cuda-oxide-*/<rev>)의 cuda-oxide/crates/rustc-codegen-cuda를 빌드해 같은 디렉터리의 임시 파일로 복사한 뒤 mv로 놓고, 그
# 체크아웃의 커밋을 source-rev.txt에 쓴다. 포크 커밋이 백엔드 크레이트의 의존 폐포를 건드리지 않은 핀 이동은 앞 rev의
# 백엔드를 같은 방법(임시 파일 + mv)으로 복사해 써도 된다. 그때는 사본의 md5와 근거를 그 디렉터리의 PROVENANCE에 적고,
# source-rev.txt는 그 .so를 지은 커밋이 아니라 그 .so가 섬기는 rev를 적는다.
OXREV=$(sed -n '/^\[patch\."https:\/\/github.com\/NVIDIA\/cuda-rust.git"\]/,/^\[workspace/s/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$HERE/Cargo.toml")
[ -n "$OXREV" ] || OXREV=$(sed -n 's/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$HERE/Cargo.toml" | head -1)
[ -n "$OXREV" ] || { echo "box.sh: no cuda-oxide rev in $HERE/Cargo.toml" >&2; exit 70; }
OXIDE="export CUDA_OXIDE_BACKEND=\$HOME/.cargo/cuda-oxide-bloomery/$OXREV/librustc_codegen_cuda.so && "
case "$*" in
  *"cargo oxide"*) OXIDE="${OXIDE}{ [ -f \"\$CUDA_OXIDE_BACKEND\" ] || { echo \"box.sh: no cuda-oxide backend for rev $OXREV at \$CUDA_OXIDE_BACKEND (tools/box.sh header)\" >&2; exit 70; }; } && { __r=\$(cat \"\${CUDA_OXIDE_BACKEND%/*}/source-rev.txt\" 2>/dev/null); [ \"\$__r\" = $OXREV ] || { echo \"box.sh: the backend at \$CUDA_OXIDE_BACKEND records source rev '\$__r', not $OXREV\" >&2; exit 70; }; unset __r; } && " ;;
esac
# serve의 build.rs가 `/props`의 version에 새기는 커밋. 박스 사본에는 .git이 없어서(rsync가 뺀다) 여기서 넘긴다.
# 이 트리에 커밋에 없는 변경이 있으면 `-dirty`를 붙인다 — 그 바이너리는 그 커밋의 것이 아니다.
COMMIT=$(git -C "$HERE" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
[ -z "$(git -C "$HERE" status --porcelain 2>/dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
ssh "$HOST" "${GUARD}source ~/bloomery-env.sh && { $PICK
} && cd $REMOTE && $V41 && $TIERX$FWD$PROFILE && $DATA && export BLOOMERY_GIT_COMMIT=$COMMIT BLOOMERY_BOX_CARD=$CARD && $OXIDE$ENVS$*"
