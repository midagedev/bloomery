#!/usr/bin/env bash
# GPU 게이트 러너 — 박스에서, box.sh가 들어간 원격 디렉터리에서 돈다. 레시피가 방금 지은
# target/release/<이름>을 카드의 게이트 락 아래에서 상한과 함께 돌리고, 그 종료 코드를
# 그대로 돌려준다. 이름 뒤의 인자는 바이너리에 그대로 붙는다.
#   ./tools/box.sh 'cargo oxide build … --bin gate_p3 && bash tools/gpu-gate.sh gate_p3'
#
# 막는 실패: 매달린 GPU 게이트 하나가 게이트 락을 무기한 쥐는 것. 락은 3090 게이트를 한 줄로 세우는 자리라,
# 한 게이트가 서면 그 락을 기다리는 트랙이 전부 선다. CPU 게이트의 900초 상한(tools/gate.sh)과 같은 값을
# 쓰고, 넘으면 빨간 게이트로 끝난다. 락과 상한의 소유자는 이 스크립트 하나다 — 레시피에 락 경로가 직접
# 나오면 check-recipes가 빨강이다.
#
# 종료 코드: 게이트의 것 그대로. 124 = 상한에서 TERM으로 끝남, 137 = TERM을 무시해 --kill-after의 KILL로
# 끝남, 75 = 30분 안에 카드 락이나 V4.1 적재 락을 못 잡음, 또는 강제한 카드(3090·a6000)가 잡힌 타이밍 임대의
# 타이밍 카드임(모두 경쟁이지 게이트 실패가 아니다), 70 = 카드를 강제했는데 임대를 테스트할 수 없음, 64 = 사용법, 2 = 바이너리 없음,
# 69 = 게이트 락이나 V4.1 적재 락 파일을 열 수 없음(박스의 root 셸이 아님).
# `--self-test`: 카드 고르기, 임대 거절, V4.1 적재 락을 임시 디렉터리의 락·임대 파일과 가짜 nvidia-smi로 시험한다(리눅스,
# flock과 timeout이 필요하다 — 맥에는 없다). 시험만 쓰는 첫 인자 `--test-locks DIR`이 세 락을 DIR 아래로 옮기고
# V4.1 적재 락의 상한을 6초로, 카드 락의 상한을 4초로, 폴링을 1초로 줄인다.
# 환경: BLOOMERY_GATE_BOUND(초, 기본 900 — tools/gate.sh와 같은 레버), BLOOMERY_GATE_CARD(아래 카드 고르기),
# BLOOMERY_BOX_CARD(box.sh가 넘기는 카드 선택 — 아래), BLOOMERY_GATE_V41_LOAD(1이면 V4.1 적재 락도 잡는다 — 아래).
# BLOOMERY_GATE_STACKS(초 — 설정하면 바이너리의 출력이 그만큼 멈출 때 tools/ref/stack-watch.sh가 스레드 스택을 뜨고 끝낸다).
set -uo pipefail
GATE_LOCK=/root/bloomery-gate.lock A6000_LOCK=/root/bloomery-gate-a6000.lock
V41_LOCK=/root/bloomery-v41-load.lock V41_BOUND=1800 POLL=5 CARD_BOUND=1800

# The runner's own tests: each case runs this script with --test-locks <tmp> (the three locks under <tmp>,
# the V4.1 load lock's bound 6 s, the card locks' 4 s, the polls 1 s), BLOOMERY_LEASE_LOCK at a file the test holds or not, a stub
# nvidia-smi on PATH (two cards, no compute process) and a stub target/release/ok that sleeps
# STUB_SLEEP seconds. Concurrent runs are started with & and waited for by their $!. One line per case;
# exit 0 iff none failed.
self_test() {
  local self t n=0 bad=0 out rc p1 t0 el
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/gpu-gate-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  mkdir -p "$t/bin" "$t/tree/target/release"
  # The stub cards carry the tree's UUIDs (tools/ref/cards.sh), the ones timing-card.sh's default names.
  local GPU_3090 GPU_A6000
  # shellcheck source=tools/ref/cards.sh
  . "$(dirname "$self")/ref/cards.sh"
  printf '%s\n' '#!/bin/sh' "case \"\$*\" in *query-gpu=*) printf '%s, NVIDIA GeForce RTX 3090\\n%s, NVIDIA RTX A6000\\n' $GPU_3090 $GPU_A6000 ;; esac" > "$t/bin/nvidia-smi"
  printf '%s\n' '#!/bin/sh' 'sleep "${STUB_SLEEP:-0}"' 'echo "ran on $CUDA_VISIBLE_DEVICES"' > "$t/tree/target/release/ok"
  chmod +x "$t/bin/nvidia-smi" "$t/tree/target/release/ok"
  # Where util-linux flock or coreutils timeout is missing (the Mac), stand-ins for the forms this runner
  # and lease-probe.sh use: `flock [-s|-x] [-n | -w S] [-E C] FD`, `flock -u FD`, `flock ... FILE CMD`
  # (flock(2) on the inherited descriptor, as util-linux), and `timeout [--kill-after=S] S CMD`.
  if ! command -v flock > /dev/null; then
    cat > "$t/bin/flock" << 'PY'
#!/usr/bin/env python3
import fcntl, os, subprocess, sys, time
a, mode, nb, wait, code = sys.argv[1:], fcntl.LOCK_EX, False, None, 1
while a and a[0].startswith("-"):
    o = a.pop(0)
    if o == "-s": mode = fcntl.LOCK_SH
    elif o == "-x": mode = fcntl.LOCK_EX
    elif o == "-u": mode = fcntl.LOCK_UN
    elif o == "-n": nb = True
    elif o == "-w": wait = float(a.pop(0))
    elif o == "-E": code = int(a.pop(0))
    else: sys.exit(f"flock stand-in: option {o}")
tgt, cmd = a[0], a[1:]
fd = int(tgt) if tgt.isdigit() and not cmd else os.open(tgt, os.O_RDWR | os.O_CREAT, 0o644)
if mode == fcntl.LOCK_UN:
    fcntl.flock(fd, mode); sys.exit(0)
end = time.monotonic() + (0 if nb else wait if wait is not None else 1e9)
while True:
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB); break
    except BlockingIOError:
        if time.monotonic() >= end: sys.exit(code)
        time.sleep(0.05)
sys.exit(subprocess.call(cmd) if cmd else 0)
PY
    chmod +x "$t/bin/flock"
  fi
  if ! command -v timeout > /dev/null; then
    printf '%s\n' '#!/usr/bin/env python3' 'import subprocess, sys' 'a = [x for x in sys.argv[1:] if not x.startswith("--kill-after")]' \
      'try: sys.exit(subprocess.run(a[1:], timeout=float(a[0])).returncode)' 'except subprocess.TimeoutExpired: sys.exit(124)' > "$t/bin/timeout"
    chmod +x "$t/bin/timeout"
  fi
  PATH=$t/bin:$PATH
  # gate <out file> <BLOOMERY_GATE_CARD> <lease file> [K=V…]: one run of the runner, its rc
  gate() {
    local o=$1 c=$2 l=$3
    shift 3
    (cd "$t/tree" && env -u BLOOMERY_BOX_CARD -u BLOOMERY_GATE_BOUND -u BLOOMERY_GATE_V41_LOAD -u BLOOMERY_GATE_STACKS -u CUDA_VISIBLE_DEVICES \
      PATH="$t/bin:$PATH" BLOOMERY_GATE_CARD="$c" BLOOMERY_LEASE_LOCK="$l" BLOOMERY_LEASE_PROC="$t/proc" "$@" \
      bash "$self" --test-locks "$t" ok) > "$o" 2>&1
  }
  # judge <name> <rc> <want rc> <ERE that must match> [ERE that must not]
  judge() {
    n=$((n + 1))
    out=$(cat "$t/out")
    if [ "$2" = "$3" ] && printf '%s\n' "$out" | grep -Eq -- "$4" && { [ -z "${5:-}" ] || ! printf '%s\n' "$out" | grep -Eq -- "$5"; }; then
      echo "ok $1"
    else
      bad=$((bad + 1))
      echo "FAIL $1: rc $2 (want $3), /$4/${5:+, not /$5/}"
      printf '%s\n' "$out" | sed 's/^/    | /'
    fi
  }
  case_() { # <name> <want rc> <ERE> <card> <lease file> [K=V…]
    local name=$1 want=$2 pat=$3
    shift 3
    rc=0
    gate "$t/out" "$@" || rc=$?
    judge "$name" "$rc" "$want" "$pat"
  }
  # The stub /proc tree ($t/proc, BLOOMERY_LEASE_PROC): a lease holder is descriptor 9 on the lease file
  # with the lock in its fdinfo; every process has a stat line with its ppid.
  # proc <pid> <ppid> [holder]
  proc() {
    mkdir -p "$t/proc/$1/fd" "$t/proc/$1/fdinfo"
    printf '%s (bash) S %s 0 0\n' "$1" "$2" > "$t/proc/$1/stat"
    if [ "${3:-}" = holder ]; then
      ln -sf "$t/lease" "$t/proc/$1/fd/9"
      printf 'pos:\t0\nflags:\t01\nlock:\t1: FLOCK  ADVISORY  WRITE %s 00:00:0 0 EOF\n' "$1" > "$t/proc/$1/fdinfo/9"
    fi
  }
  record() { rm -rf "$t/lease.card"; printf 'pid=%s timing_gpu=%s\n' "$1" "$2" > "$t/lease.card"; }
  rm -rf "$t/proc"
  mkdir -p "$t/proc"
  : > "$t/lease"
  exec 7> "$t/lease"
  flock -x 7 || { echo "FAIL: the test cannot hold its lease file"; return 1; }
  # The runner 100 (ppid 1) whose child 101 holds the lease (descriptor 9 inherited); 300 holds nothing.
  proc 100 1
  proc 101 100 holder
  proc 300 1
  local NOREC='the lease holds no timing-card record'
  case_ 'lease held, no record: 3090 forced is refused, 75, named' 75 "BLOOMERY_GATE_CARD=3090 .*in doubt \\($NOREC \\(no readable .*a tree without the writer" 3090 "$t/lease"
  case_ '  … a6000 forced is refused too' 75 "BLOOMERY_GATE_CARD=a6000 .*in doubt \\($NOREC" a6000 "$t/lease"
  case_ '  … any waits on both cards (75 at the bound)' 75 'no gate lock \(any\) was free within 4 s' any "$t/lease"
  record 101 "$GPU_3090"
  case_ 'the holder records the 3090: 3090 forced is refused, 75, named' 75 'BLOOMERY_GATE_CARD=3090 .*held by a sitting that times the 3090' 3090 "$t/lease"
  case_ '  … a6000 forced runs on the A6000' 0 "ran on $GPU_A6000" a6000 "$t/lease"
  case_ '  … any takes the A6000' 0 'ok on the a6000 \(asked any\)' any "$t/lease"
  exec 6> "$t/gate-a6000.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the A6000 gate lock"; return 1; }
  case_ '  … any with the A6000 taken waits, never the 3090 (75 at the bound)' 75 'no gate lock \(any\) was free within 4 s' any "$t/lease"
  flock -u 6
  exec 6>&-
  record 100 "$GPU_A6000"
  case_ "the holder's ancestor records the A6000: a6000 forced is refused, 75, named" 75 'BLOOMERY_GATE_CARD=a6000 .*held by a sitting that times the a6000' a6000 "$t/lease"
  case_ '  … 3090 forced runs' 0 'ran on $' 3090 "$t/lease"
  case_ '  … any takes the 3090' 0 'ok on the 3090 \(asked any\)' any "$t/lease"
  record 100 none
  case_ 'the record says none (a CPU runner): 3090 forced runs' 0 'ran on $' 3090 "$t/lease"
  case_ '  … a6000 forced runs' 0 "ran on $GPU_A6000" a6000 "$t/lease"
  case_ '  … any takes the A6000' 0 'ok on the a6000 \(asked any\)' any "$t/lease"
  record 999 "$GPU_A6000"
  case_ 'a stale record (its pid gone): 3090 forced is refused, named' 75 "in doubt \\($NOREC \\(.*names pid 999, which neither holds the lease nor is an ancestor" 3090 "$t/lease"
  record 300 "$GPU_A6000"
  case_ 'a foreign record (a live pid that holds nothing): 3090 forced is refused, named' 75 "in doubt \\($NOREC \\(.*names pid 300, which neither holds" 3090 "$t/lease"
  record 101 GPU-not-a-card
  case_ 'a record naming neither card: 3090 forced is refused, named' 75 "in doubt \\(.*names timing_gpu 'GPU-not-a-card', neither card" 3090 "$t/lease"
  record 101 x
  printf 'timing_gpu=%s\n' "$GPU_A6000" > "$t/lease.card"
  case_ 'a record with no pid: 3090 forced is refused, named' 75 "in doubt \\($NOREC \\(.*names no pid" 3090 "$t/lease"
  rm -f "$t/lease.card"
  mkdir "$t/lease.card"
  case_ 'an unreadable record: 3090 forced is refused, named' 75 "in doubt \\($NOREC \\(no readable" 3090 "$t/lease"
  rm -rf "$t/lease.card" "$t/proc/101/fd/9"
  record 100 "$GPU_A6000"
  case_ 'no holder readable: 3090 forced is refused, named' 75 "in doubt \\($NOREC that can be checked \\(no process holding" 3090 "$t/lease"
  flock -u 7
  exec 7>&-
  record 101 "$GPU_3090"
  case_ 'lease free: the record is not read, 3090 forced runs' 0 'ran on $' 3090 "$t/lease"
  rm -rf "$t/lease.card" "$t/proc"
  mkdir -p "$t/proc"
  case_ 'a6000 forced, lease free: the A6000' 0 "ran on $GPU_A6000" a6000 "$t/lease"
  case_ 'a6000 forced, lease cannot be tested: 70, named' 70 'cannot be tested' a6000 "$t/no-dir/lease"
  case_ 'a V4.1 load alone takes the V4.1 load lock and does not wait' 0 'ok on the 3090 \(asked 3090\), V4\.1 load lock' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  case_ 'BLOOMERY_GATE_V41_LOAD other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_V41_LOAD is 1' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=yes
  case_ 'BLOOMERY_GATE_STACKS not whole seconds: 64, named' 64 'BLOOMERY_GATE_STACKS is whole seconds' 3090 "$t/lease" BLOOMERY_GATE_STACKS=soon
  case_ 'BLOOMERY_GATE_STACKS set: the binary runs under the watch, its output through' 0 'ran on $' 3090 "$t/lease" BLOOMERY_GATE_STACKS=5
  # Two V4.1 loads at once, on the two cards: the second waits for the first, names it, and counts the wait.
  gate "$t/first" a6000 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 STUB_SLEEP=3 &
  p1=$!
  sleep 1
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_LEASE_PROC=/proc || rc=$?
  cat "$t/first" >> "$t/out"
  judge 'two V4.1 loads at once: the second waits, names the holder, counts the wait' "$rc" 0 \
    'waits for the V4\.1 load lock .*holding the 3090 gate lock'
  if [ -d /proc/self ]; then
    judge '  … its wait line names the first run as the holder' "$rc" 0 '\[lease\] +pid [0-9]+ holds it'
  else
    echo "skip   … its wait line names the first run as the holder: no /proc here (lease_holders reads it)"
  fi
  judge '  … and the waited line the batch runner subtracts' "$rc" 0 '^gpu-gate\.sh: waited [0-9]+ s for the V4\.1 load lock$'
  rc=0
  wait "$p1" || rc=$?
  cp "$t/first" "$t/out"
  judge '  … the first ran on the A6000 without waiting' "$rc" 0 "ran on $GPU_A6000" 'waits for the V4\.1'
  # A run that is not a V4.1 load does not wait for a V4.1 load on the other card.
  gate "$t/first" a6000 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 STUB_SLEEP=3 &
  p1=$!
  sleep 1
  rc=0 t0=$SECONDS
  gate "$t/out" 3090 "$t/lease" || rc=$?
  el=$((SECONDS - t0))
  echo "elapsed ${el} s" >> "$t/out"
  judge 'a run that is not a V4.1 load does not wait for one' "$rc" 0 'elapsed [0-2] s' 'V4\.1'
  wait "$p1" || true
  # The lock order: a V4.1 load that waits for its card lock holds no V4.1 load lock meanwhile, so a V4.1
  # load on the other card goes ahead.
  exec 6> "$t/gate.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the 3090 gate lock"; return 1; }
  gate "$t/first" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 &
  p1=$!
  sleep 1
  rc=0 t0=$SECONDS
  gate "$t/out" a6000 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 || rc=$?
  el=$((SECONDS - t0))
  echo "elapsed ${el} s" >> "$t/out"
  judge 'lock order: a V4.1 load queued on its card lock holds no V4.1 load lock' "$rc" 0 'elapsed [0-2] s' 'waits for the V4\.1'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  cp "$t/first" "$t/out"
  judge '  … and runs once its card lock frees' "$rc" 0 'waited [0-9]+ s for the gate lock \(3090\)'
  # The bound: a lock that never frees is 75, named, after V41_BOUND (6 s under --test-locks).
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  case_ 'the V4.1 load lock never frees: 75 after the bound, named' 75 'V4\.1 load lock .* was not free within 6 s' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  flock -u 6
  exec 6>&-
  echo "gpu-gate self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi
if [ "${1:-}" = --test-locks ]; then
  [ $# -ge 3 ] && [ -d "$2" ] || { echo "usage: gpu-gate.sh --test-locks <dir> <binary> [args...] (the self-test's)" >&2; exit 64; }
  GATE_LOCK=$2/gate.lock A6000_LOCK=$2/gate-a6000.lock V41_LOCK=$2/v41-load.lock V41_BOUND=6 POLL=1 CARD_BOUND=4
  shift 2
fi
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: gpu-gate.sh <target/release binary> [args...]" >&2; exit 64; }
shift
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/gate-bound.sh"
gate_bound gpu-gate.sh || exit $?
EXE=./target/release/$NAME
[ -x "$EXE" ] || { echo "gpu-gate.sh: no $EXE — the recipe builds it before calling this" >&2; exit 2; }
# BLOOMERY_GATE_V41_LOAD=1: the run loads the whole V4.1 model (the recipe carries [group('v41-load')] and
# exports it), so it also takes the box-wide V4.1 load lock below. 0 or unset: it does not.
V41=${BLOOMERY_GATE_V41_LOAD:-0}
case $V41 in
  0 | 1) ;;
  *) echo "gpu-gate.sh: BLOOMERY_GATE_V41_LOAD is 1 (the run loads V4.1) or 0/unset, got '$V41'" >&2; exit 64 ;;
esac
# BLOOMERY_GATE_STACKS=<seconds>: the binary runs under tools/ref/stack-watch.sh, which dumps its thread
# stacks and ends it once its output has stopped for that long, so a hang says where before the bound.
STACKS=${BLOOMERY_GATE_STACKS:-}
case $STACKS in
  '') ;;
  0 | *[!0-9]*) echo "gpu-gate.sh: BLOOMERY_GATE_STACKS is whole seconds from 1 (the quiet time before a stack dump) or unset, got '$STACKS'" >&2; exit 64 ;;
esac
# 카드 고르기. 락은 카드마다 하나: 3090은 예전 경로 그대로(돌고 있는 트랙의 옛 사본이 그 경로를 잡는다),
# A6000은 새 파일. BLOOMERY_GATE_CARD=3090(기본 — 예전과 같다) | a6000 | any. any는 A6000을 먼저 본다 —
# V4.1 `--place gate` 게이트는 3090에서만 돌 수 있으니 떠도는 게이트가 3090을 비워 두는 편이 낫다. any는
# 잡힌 타이밍 임대가 재는 카드(아래 timing_card)를 건너뛰고, A6000은 컴퓨트 프로세스가 있어도 건너뛴다. 카드를
# 강제한 실행은 그 카드가 잡힌 임대의 타이밍 카드면 75로 끝난다. 임대 탐침은
# lease-probe.sh의 lease_free(공유 잠금) 하나다 — 대기 중 5초마다 도는 이 탐침이 배타 잠금이면 다른 탐침과 부딪혀
# 빈 임대를 잡힌 것으로 읽고, 테스트할 수 없는 임대도 비었다고 읽지 않는다.
# shellcheck source=tools/ref/lease-probe.sh
source "${BASH_SOURCE[0]%/*}/ref/lease-probe.sh"
# box.sh가 보여 준 카드가 락을 정한다. BLOOMERY_BOX_CARD는 box.sh가 박스 쪽에 넘기는 그 선택이다(3090 = env
# 파일의 핀, a6000, both = 두 카드). box.sh를 거치지 않은 실행은 핀 그대로로 읽는다. box.sh가 BLOOMERY_CARD=a6000|both로
# 카드를 골랐으면 BLOOMERY_GATE_CARD 없이 그 카드의 락을 잡고 — both는 두 락을 함께 잡거나 하나도 안 잡는다 — 다른 카드를
# 이름 대면 거절한다(64). 두 카드를 보는 실행이 3090 락 하나만 쥐면 다른 트랙의 any 게이트가 비어 보이는 A6000에 올라가
# 같은 카드 메모리를 나눠 쓴다.
BOXC=${BLOOMERY_BOX_CARD:-3090}
case "$BOXC" in
  3090) CARD=${BLOOMERY_GATE_CARD:-3090} ;;
  a6000 | both)
    CARD=${BLOOMERY_GATE_CARD:-$BOXC}
    [ "$CARD" = "$BOXC" ] || {
      echo "gpu-gate.sh: box.sh put $([ "$BOXC" = both ] && echo 'both cards' || echo 'the A6000') in view (BLOOMERY_CARD=$BOXC), and BLOOMERY_GATE_CARD=$CARD names another lock — leave it unset or set it to $BOXC" >&2
      exit 64
    } ;;
  *) echo "gpu-gate.sh: BLOOMERY_BOX_CARD is 3090, a6000 or both (box.sh sets it), got '$BOXC'" >&2; exit 64 ;;
esac
case "$CARD" in
  3090 | a6000 | any) ;;
  both)
    [ "$BOXC" = both ] || {
      echo "gpu-gate.sh: BLOOMERY_GATE_CARD=both holds both cards' locks for a run that sees both — run it under box.sh's BLOOMERY_CARD=both" >&2
      exit 64
    } ;;
  *) echo "gpu-gate.sh: BLOOMERY_GATE_CARD is 3090, a6000, any or both, got '$CARD'" >&2; exit 64 ;;
esac
uuid_of() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "$1" | cut -d, -f1 | head -1; }
a6000_idle() {
  local u; u=$(uuid_of A6000); [ -n "$u" ] || return 1
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"
}
# 락 파일을 못 여는 셸(박스의 root가 아니거나 박스가 아님)은 경쟁이 아니다 — 여기서 이름을 대고 끝난다.
for lock in "9:$GATE_LOCK" "8:$A6000_LOCK"; do
  eval "exec ${lock%%:*}>${lock#*:}" 2> /dev/null || { echo "gpu-gate.sh: cannot open ${lock#*:} — the gate runs on the box, as root" >&2; exit 69; }
done
# The held lease's timing card: the record lease_take writes beside the lock once it holds it
# (tools/ref/lease.sh), `pid=<the runner> timing_gpu=<card UUID | none>`. It counts only while its pid is
# a process that holds the lease or an ancestor of one (a descriptor on LEASE_LOCK carrying the lock, the
# match lease_holders makes, and the ppid chains above them; BLOOMERY_LEASE_PROC names another /proc tree
# for the stub tests). Prints 3090, a6000, none (the runner times no GPU) or `doubt: <why>`: a missing,
# stale, foreign or unreadable record is doubt — a runner from a tree without the writer — and doubt is
# never read as a free card or as timing-card.sh's default.
timing_card() {
  local rec=$LEASE_LOCK.card f pid='' gpu='' chain u3 ua
  local none="the lease holds no timing-card record"
  if [ ! -f "$rec" ] || [ ! -r "$rec" ]; then
    echo "doubt: $none (no readable $rec: a runner from a tree without the writer)"
    return
  fi
  for f in $(head -1 "$rec"); do
    case $f in pid=*) pid=${f#pid=} ;; timing_gpu=*) gpu=${f#timing_gpu=} ;; esac
  done
  case $pid in '' | *[!0-9]*)
    echo "doubt: $none ($rec names no pid)"
    return ;;
  esac
  if ! chain=$(python3 - "$LEASE_LOCK" "${BLOOMERY_LEASE_PROC:-/proc}" 2>&1 << 'PY'
import os, sys
lock, proc = sys.argv[1], sys.argv[2]
try:
    want = os.stat(lock)
    names = os.listdir(proc)
except OSError as e:
    sys.exit(f"{e.filename} cannot be read ({e.strerror})")


def ppid(pid):
    try:
        stat = open(f"{proc}/{pid}/stat").read()
        return int(stat[stat.rindex(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


out = set()
for name in names:
    if not name.isdigit():
        continue
    try:
        fds = os.listdir(f"{proc}/{name}/fd")
    except OSError:
        continue
    for fd in fds:
        try:
            st = os.stat(f"{proc}/{name}/fd/{fd}")
            info = open(f"{proc}/{name}/fdinfo/{fd}").read()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (want.st_dev, want.st_ino) and any(
                l.startswith("lock:") and "->" not in l for l in info.split("\n")):
            p = int(name)
            while p > 1 and p not in out:
                out.add(p)
                p = ppid(p)
            break
if not out:
    sys.exit(f"no process holding {lock} could be read in {proc}")
print("\n".join(str(p) for p in sorted(out)))
PY
  ); then
    echo "doubt: $none that can be checked (${chain//$'\n'/ })"
    return
  fi
  if ! grep -qx "$pid" <<< "$chain"; then
    echo "doubt: $none ($rec names pid $pid, which neither holds the lease nor is an ancestor of a holder: stale or foreign)"
    return
  fi
  u3=$(uuid_of 3090) ua=$(uuid_of A6000)
  if [ "$gpu" = none ]; then
    echo none
  elif [ -n "$gpu" ] && [ "$gpu" = "$u3" ]; then
    echo 3090
  elif [ -n "$gpu" ] && [ "$gpu" = "$ua" ]; then
    echo a6000
  else
    echo "doubt: $rec names timing_gpu '$gpu', neither card (3090 '$u3', A6000 '$ua')"
  fi
}
# sitting_on <card>: 0 when the timing lease is held and times <card>, or its timing card is in doubt; 1
# when it is free or times the other card; 2 when the lease cannot be tested (lease_free named it).
# SITTING_WHY names a doubt for the refusal.
SITTING_WHY=
sitting_on() {
  local lrc=0 tc
  lease_free || lrc=$?
  case $lrc in
    0) return 1 ;;
    1)
      tc=$(timing_card)
      case $tc in
        "$1") SITTING_WHY="a sitting that times the $1"; return 0 ;;
        none) return 1 ;;
        doubt:*) SITTING_WHY="a sitting whose timing card is in doubt (${tc#doubt: }), read as the $1"; return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) return 2 ;;
  esac
}
# A card forced onto the sitting's timing card does not wait the sitting out beside it: 75 at once,
# named (box.sh's guard waits for the sitting before the next try); an untestable lease is 70. `any`
# passes the timing card by for the other one instead, and the A6000 also when it has a compute process.
refuse_forced() {
  local src=0
  sitting_on "$1" || src=$?
  case $src in
    0) echo "gpu-gate.sh: BLOOMERY_GATE_CARD=$1 and the timing lease $LEASE_LOCK is held by $SITTING_WHY: not starting on it — contention, not a red gate (rc 75)" >&2; exit 75 ;;
    1) ;;
    *) echo "gpu-gate.sh: BLOOMERY_GATE_CARD=$1 and the timing lease $LEASE_LOCK cannot be tested (above): not starting (rc 70)" >&2; exit 70 ;;
  esac
}
# Under `any` a card in doubt, or an untestable lease, is passed by like the timing card.
take_a6000() {
  local src=0
  flock -n 8 || return 1
  if [ "$CARD" = a6000 ]; then
    refuse_forced a6000
  elif [ "$CARD" = any ]; then
    sitting_on a6000 || src=$?
    if [ "$src" != 1 ] || ! a6000_idle; then
      flock -u 8
      return 1
    fi
  fi
  GOT=a6000
}
take_3090() {
  local src=0
  flock -n 9 || return 1
  if [ "$CARD" = 3090 ]; then
    refuse_forced 3090
  elif [ "$CARD" = any ]; then
    sitting_on 3090 || src=$?
    if [ "$src" != 1 ]; then
      flock -u 9
      return 1
    fi
  fi
  GOT=3090
}
GOT=
LOCK_T0=$SECONDS
if [ "$CARD" = both ]; then
  # 3090 락을 먼저, 그것을 쥔 채 A6000 락을. 다른 실행은 락을 하나만 쥐므로 이 순서로는 교착이 없고, 막혀 기다리는
  # flock은 해제 순간 깨어나 5초마다 도는 한 카드 실행들의 폴링보다 먼저 잡는다(둘이 함께 빌 때만 잡는 폴링은 게이트가
  # 이어지는 동안 굶는다).
  start=$SECONDS
  if flock -w 1800 9; then
    left=$((1800 - (SECONDS - start)))
    if flock -w "$((left > 0 ? left : 1))" 8; then GOT=both; else flock -u 9; fi
  fi
else
  for ((waited = 0; waited <= CARD_BOUND; waited += POLL)); do
    case "$CARD" in
      3090) take_3090 ;;
      a6000) take_a6000 ;;
      any) take_a6000 || take_3090 ;;
    esac
    [ -n "$GOT" ] && break
    sleep "$POLL"
  done
fi
if [ -z "$GOT" ]; then
  echo "gpu-gate.sh: no gate lock ($CARD) was free within ${CARD_BOUND} s — contention, not a red gate" >&2
  exit 75
fi
# The wait is not the gate's time: tools/gate-batch.sh subtracts this line's seconds from the item's times row.
LOCK_WAIT=$((SECONDS - LOCK_T0))
[ "$LOCK_WAIT" = 0 ] || echo "gpu-gate.sh: waited ${LOCK_WAIT} s for the gate lock ($CARD)" >&2
# The V4.1 load lock: one V4.1 load on the box at a time, across trees and batches — two at once push each
# other's host set (~190 GB of the 256 GB) out of the page cache and both turn IO-bound. Lock order, the
# deadlock rule: it is taken only while this run already holds its card lock(s), always after them, and
# no card lock is ever taken while holding it; a run that is not a V4.1 load never touches it. So a run
# holding it waits on nothing, and no cycle can form. The wait holds the card lock(s) — that card is the
# one this load will use — and names the lock's holders at once and once a minute; after V41_BOUND
# seconds it is 75 (contention, not a red gate). The descriptor stays open for the binary, so the lock
# lives exactly as long as the load's process.
if [ "$V41" = 1 ]; then
  eval "exec 7>$V41_LOCK" 2> /dev/null || { echo "gpu-gate.sh: cannot open $V41_LOCK — the gate runs on the box, as root" >&2; exit 69; }
  V41_T0=$SECONDS said=-60
  until flock -n 7; do
    el=$((SECONDS - V41_T0))
    if [ "$el" -ge "$V41_BOUND" ]; then
      echo "gpu-gate.sh: $NAME: the V4.1 load lock $V41_LOCK was not free within ${V41_BOUND} s — contention, not a red gate (rc 75)" >&2
      exit 75
    fi
    if [ $((el - said)) -ge 60 ]; then
      echo "gpu-gate.sh: $NAME waits for the V4.1 load lock $V41_LOCK (another run is loading V4.1), holding the $GOT gate lock; ${el} s so far, its holders:" >&2
      lease_holders "$V41_LOCK" >&2
      said=$el
    fi
    sleep "$POLL"
  done
  # Not the gate's time either: tools/gate-batch.sh subtracts this line's seconds as it does the card lock's.
  V41_WAIT=$((SECONDS - V41_T0))
  [ "$V41_WAIT" = 0 ] || echo "gpu-gate.sh: waited ${V41_WAIT} s for the V4.1 load lock" >&2
fi
if [ "$GOT" = a6000 ] || [ "$CARD" = any ]; then
  U=$(uuid_of "$([ "$GOT" = a6000 ] && echo A6000 || echo 3090)")
  [ -n "$U" ] || { echo "gpu-gate.sh: the $GOT lookup failed" >&2; exit 75; }
  export CUDA_VISIBLE_DEVICES=$U
fi
echo "gpu-gate.sh: $NAME on $([ "$GOT" = both ] && echo 'both cards, both gate locks' || echo "the $GOT") (asked $CARD)$([ "$V41" = 0 ] || echo ', V4.1 load lock')" >&2
if [ -n "$STACKS" ]; then
  bash "${BASH_SOURCE[0]%/*}/ref/stack-watch.sh" "$STACKS" "$NAME" -- timeout --kill-after=10 "$BOUND" "$EXE" "$@"
else
  timeout --kill-after=10 "$BOUND" "$EXE" "$@"
fi
rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GPU GATE TIMED OUT: $NAME after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GPU GATE RED: $NAME (exit $rc)" >&2
fi
exit "$rc"
