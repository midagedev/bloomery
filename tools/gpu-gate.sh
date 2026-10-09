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
# 끝남, 75 = 30분 안에 카드 락이나 V4.1 적재 락을 못 잡음(카드 락 상한에는 배치 홀드를 기다린 시간이 든다), 또는 강제한 카드(3090·a6000)가 잡힌 타이밍 임대의
# 타이밍 카드임(모두 경쟁이지 게이트 실패가 아니다), 70 = 카드를 강제했는데 임대를 테스트할 수 없음, 64 = 사용법, 2 = 바이너리 없음,
# 69 = 게이트 락이나 V4.1 적재 락 파일을 열 수 없음(박스의 root 셸이 아님).
# `--self-test`: 카드 고르기, 임대 거절, V4.1 적재 락을 임시 디렉터리의 락·임대 파일과 가짜 nvidia-smi로 시험한다(리눅스,
# flock과 timeout이 필요하다 — 맥에는 없다). 시험만 쓰는 첫 인자 `--test-locks DIR`이 세 락과 배치 홀드 파일(batch.gpuhold)을 DIR 아래로 옮기고
# V4.1 적재 락의 상한을 6초로, 카드 락의 상한을 4초로, 폴링을 1초로 줄인다.
# 환경: BLOOMERY_GATE_BOUND(초, 기본 900 — tools/gate.sh와 같은 레버), BLOOMERY_GATE_CARD(아래 카드 고르기),
# BLOOMERY_BOX_CARD(box.sh가 넘기는 카드 선택 — 아래), BLOOMERY_GATE_V41_LOAD(1이면 V4.1 적재 락도 잡는다 — 아래).
# BLOOMERY_BATCH_OWNER(letters, digits, _: the owner of the batch hold this run's batch put up — the batch hold paragraph below).
# BLOOMERY_GATE_STACKS(초 — 설정하면 바이너리의 출력이 그만큼 멈출 때 tools/ref/stack-watch.sh가 스레드 스택을 뜨고 끝낸다).
# BLOOMERY_GATE_GDB=1: the binary runs under gdb, which prints every thread's stack when a signal ends it.
# BLOOMERY_TIER=real|fixture (unset: real) and BLOOMERY_FIXTURE_MODEL: the fixture tier (tools/box.sh resolves both, see
# tools/ref/ref-paths.sh). A run asking for the V4.1 load lock takes none when its tier is fixture AND its
# BLOOMERY_REF_MODEL is the file box.sh resolved the tier to (BLOOMERY_FIXTURE_MODEL): the lock keeps two ~190 GB host sets
# out of each other's page cache, and a fixture's host set is a few GB. The request alone does not skip it: a fixture tier whose
# model is not the resolved fixture file (a caller's own BLOOMERY_REF_MODEL, a family the tier left on its real file) takes the
# lock as a real load does, and says so. A fixture run does not open the lock file at all. Any other BLOOMERY_TIER value is 64.
# Locks. A run takes its card lock(s) and, when it loads, the V4.1 load lock TOGETHER: each poll tries every lock it needs
# without blocking, and a run that got the card but not the load lock lets the card go before it sleeps, so a loader queued behind
# another load holds no card — an `any` gate takes it. Nothing waits while holding a lock, except the two-card order below.
# The batch hold. While the lead's landing batch (tools/gate-batch.sh --ledger) runs, the file /root/bloomery-batch.gpuhold is up on
# the box: `owner=<id> since=<epoch>`, its mtime refreshed by the batch's heartbeat. A run whose BLOOMERY_BATCH_OWNER (which the
# batch puts in each of its items' box env) is not that owner WAITS at the top of every poll — before the V4.1 load lock, before any
# card lock, so it holds none — printing `[batch-hold]` lines at once and once a minute, inside CARD_BOUND (75 at the bound, as
# contention is today, with its seconds in a `waited N s for the batch hold` line that tools/gate-batch.sh subtracts). The batch's own
# items pass. A hold whose mtime is 300 s old is stale (the batch or its link to the box is gone): the run ignores it, saying so once.
# A hold that cannot be read is read as up, never as free. The name does not match tools/ref/lease-probe.sh's LEASE_HOLDS
# (/root/bloomery-*-hold): builds and every non-GPU box command ignore it. The file has one writer, this script:
#   gpu-gate.sh --gpuhold up|beat|down <owner>   up: put it up (75 while another owner's fresh hold is up; a stale one is taken over);
#                                                beat: refresh its mtime, never create it (3: gone or another owner's);
#                                                down: remove it, only if it is <owner>'s; (owner: letters, digits, _)
#   gpu-gate.sh --gpuhold path                   print the path
# The batch calls these through tools/box.sh with BLOOMERY_BOX_READONLY=1 (no sync, no guard wait: a heartbeat stuck behind a sitting
# would go stale), a deliberate write of this one control file.
set -uo pipefail
GATE_LOCK=/root/bloomery-gate.lock A6000_LOCK=/root/bloomery-gate-a6000.lock
V41_LOCK=/root/bloomery-v41-load.lock V41_BOUND=1800 POLL=1 CARD_BOUND=1800
GPU_HOLD=/root/bloomery-batch.gpuhold HOLD_STALE=300
# shellcheck source=tools/ref/lease-probe.sh
source "${BASH_SOURCE[0]%/*}/ref/lease-probe.sh"

# batch_hold_read: HOLD_STATE none | up | stale (its mtime is HOLD_STALE s old or more), HOLD_OWNER (`?` when the file names none or
# is not readable), HOLD_AGE (seconds since its last refresh, `?` when its mtime cannot be read — then it is up: never free) and
# HOLD_SINCE (the epoch the batch put it up, empty when unreadable). A time in the future (a stepped clock) reads as age 0.
HOLD_STATE=none HOLD_OWNER='' HOLD_AGE=0 HOLD_SINCE='' HOLD_STALE_SAID=0
batch_hold_read() {
  local m line f
  HOLD_STATE=none HOLD_OWNER='' HOLD_AGE=0 HOLD_SINCE=''
  [ -e "$GPU_HOLD" ] || return 0
  HOLD_OWNER='?'
  line=$(head -1 "$GPU_HOLD" 2> /dev/null) || line=
  set -f
  for f in $line; do
    case $f in
      owner=*) HOLD_OWNER=${f#owner=} ;;
      since=*[!0-9]*) ;;
      since=?*) HOLD_SINCE=${f#since=} ;;
    esac
  done
  set +f
  case $HOLD_OWNER in '' | *[!A-Za-z0-9_]*) HOLD_OWNER='?' ;; esac
  m=$(__lease_mtime "$GPU_HOLD")
  case $m in
    '' | *[!0-9]*) HOLD_STATE=up HOLD_AGE='?'; return 0 ;;
  esac
  HOLD_AGE=$(($(date +%s) - m))
  [ "$HOLD_AGE" -ge 0 ] || HOLD_AGE=0
  if [ "$HOLD_AGE" -ge "$HOLD_STALE" ]; then HOLD_STATE=stale; else HOLD_STATE=up; fi
}

# batch_hold_blocks: 0 when a fresh hold of another owner is up (the run waits), else 1. A stale hold is named once, then ignored.
batch_hold_blocks() {
  batch_hold_read
  case $HOLD_STATE in
    up) [ "$HOLD_OWNER" != "${BLOOMERY_BATCH_OWNER:-}" ] ;;
    stale)
      if [ "$HOLD_STALE_SAID" = 0 ]; then
        echo "[batch-hold] $(now) ${NAME:-gpu-gate.sh}: the batch hold $GPU_HOLD (owner $HOLD_OWNER) is stale — not refreshed for $HOLD_AGE s (stale at $HOLD_STALE s): its batch is gone or cut off from the box; ignoring it" >&2
        HOLD_STALE_SAID=1
      fi
      return 1 ;;
    *) return 1 ;;
  esac
}

# gpuhold <verb> <owner>: the hold file's one writer (the header). Prints what it did; the rc is the verb's.
gpuhold() {
  local verb=${1:-} owner=${2:-} tmp
  if [ "$verb" = path ]; then echo "$GPU_HOLD"; return 0; fi
  case $verb in
    up | beat | down) ;;
    *) echo "gpu-gate.sh: --gpuhold takes path or up|beat|down <owner>, got '$verb'" >&2; return 64 ;;
  esac
  case $owner in
    '' | *[!A-Za-z0-9_]*) echo "gpu-gate.sh: --gpuhold $verb: an owner is letters, digits and _, got '$owner'" >&2; return 64 ;;
  esac
  batch_hold_read
  case $verb in
    up)
      case $HOLD_STATE in
        up)
          if [ "$HOLD_OWNER" != "$owner" ]; then
            echo "gpu-gate.sh: the batch hold $GPU_HOLD is up for owner $HOLD_OWNER (refreshed $HOLD_AGE s ago): not taking it (rc 75)" >&2
            return 75
          fi ;;
        stale)
          echo "gpu-gate.sh: the batch hold $GPU_HOLD of owner $HOLD_OWNER was not refreshed for $HOLD_AGE s: taking it over" >&2
          rm -f "$GPU_HOLD" ;;
      esac
      tmp=$GPU_HOLD.$$.new
      printf 'owner=%s since=%s\n' "$owner" "$(date +%s)" > "$tmp" 2> /dev/null || { echo "gpu-gate.sh: cannot write $tmp — the hold is written on the box, as root" >&2; return 69; }
      if [ "$HOLD_STATE" = up ]; then
        mv -f "$tmp" "$GPU_HOLD"
      elif ln "$tmp" "$GPU_HOLD" 2> /dev/null; then
        rm -f "$tmp"
      else
        rm -f "$tmp"
        echo "gpu-gate.sh: another batch put $GPU_HOLD up at the same moment: not taking it (rc 75)" >&2
        return 75
      fi
      echo "gpu-gate.sh: batch hold $GPU_HOLD up: owner $owner" ;;
    beat)
      if [ "$HOLD_STATE" = none ] || [ "$HOLD_OWNER" != "$owner" ]; then
        echo "gpu-gate.sh: the batch hold $GPU_HOLD is $([ "$HOLD_STATE" = none ] && echo gone || echo "owner $HOLD_OWNER's"), not $owner's: not refreshing it (rc 3)" >&2
        return 3
      fi
      touch -c "$GPU_HOLD" ;;
    down)
      if [ "$HOLD_STATE" = none ]; then
        echo "gpu-gate.sh: the batch hold $GPU_HOLD is already down"
      elif [ "$HOLD_OWNER" != "$owner" ]; then
        echo "gpu-gate.sh: the batch hold $GPU_HOLD is owner $HOLD_OWNER's, not $owner's: left up"
      else
        rm -f "$GPU_HOLD"
        echo "gpu-gate.sh: batch hold $GPU_HOLD down: owner $owner"
      fi ;;
  esac
}

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
  # The stub nvidia-smi is written before the tree's cards.sh is sourced (below, once PATH holds
  # it), so the resolution reads made-up UUIDs (depth-stub-cards.sh's, never the box's) and the
  # self-test needs no card: name,uuid is cards.sh's query, uuid,name uuid_of's further down.
  printf '%s\n' '#!/bin/sh' 'case "$*" in' \
    '  *query-gpu=name,uuid*) printf "NVIDIA GeForce RTX 3090, GPU-11111111-1111-1111-1111-111111111111\nNVIDIA RTX A6000, GPU-00000000-0000-0000-0000-000000000000\n" ;;' \
    '  *query-gpu=*) printf "GPU-11111111-1111-1111-1111-111111111111, NVIDIA GeForce RTX 3090\nGPU-00000000-0000-0000-0000-000000000000, NVIDIA RTX A6000\n" ;;' \
    'esac' > "$t/bin/nvidia-smi"
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
  # The tree's cards resolve through the stub above: timing-card.sh's two names are what it
  # answers, and the UUIDs the cases below compare against are the stub's.
  local GPU_3090 GPU_A6000
  # shellcheck source=tools/ref/cards.sh
  . "$(dirname "$self")/ref/cards.sh"
  { [ -n "$GPU_3090" ] && [ -n "$GPU_A6000" ]; } || {
    echo "FAIL: the self-test's stub cards did not resolve (CARDS_ERROR: ${CARDS_ERROR:-none})"
    return 1
  }
  # gate <out file> <BLOOMERY_GATE_CARD> <lease file> [K=V…]: one run of the runner, its rc (TL=<dir>: the dir its locks and its
  # batch hold live in, default the test's own)
  gate() {
    local o=$1 c=$2 l=$3
    shift 3
    # Under a bound of its own: a runner that hangs ends the case red (124), it does not hang the check.
    (cd "$t/tree" && timeout --kill-after=5 60 env -u BLOOMERY_BOX_CARD -u BLOOMERY_GATE_BOUND -u BLOOMERY_GATE_V41_LOAD -u BLOOMERY_GATE_STACKS -u BLOOMERY_GATE_GDB -u CUDA_VISIBLE_DEVICES \
      -u BLOOMERY_TIER -u BLOOMERY_FIXTURE_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_BATCH_OWNER \
      PATH="$t/bin:$PATH" BLOOMERY_GATE_CARD="$c" BLOOMERY_LEASE_LOCK="$l" BLOOMERY_LEASE_PROC="$t/proc" "$@" \
      bash "$self" --test-locks "${TL:-$t}" ok) > "$o" 2>&1
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
  case_ "a run closes with its own wall and exit" 0 '^gpu-gate\.sh: ok ran [0-9]+ s \(exit 0\)$' 3090 "$t/lease"
  case_ 'BLOOMERY_GATE_GDB other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_GDB is 1' 3090 "$t/lease" BLOOMERY_GATE_GDB=yes
  case_ 'BLOOMERY_GATE_GDB with BLOOMERY_GATE_STACKS: 64, named' 64 'set one' 3090 "$t/lease" BLOOMERY_GATE_GDB=1 BLOOMERY_GATE_STACKS=5
  # Two V4.1 loads at once, on the two cards: the second waits for the first, names it, and counts the wait.
  gate "$t/first" a6000 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 STUB_SLEEP=3 &
  p1=$!
  sleep 1
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_LEASE_PROC=/proc || rc=$?
  cat "$t/first" >> "$t/out"
  judge 'two V4.1 loads at once: the second waits, names the holder, counts the wait' "$rc" 0 \
    'waits for the V4\.1 load lock .*holding no gate lock'
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
  # Half a poll later, so the second run's first look at the load lock falls between the first run's polls: each poll of a queued
  # loader tests the lock with a shared hold of a few ms, and a second loader's exclusive try that lands inside it is refused once.
  sleep 1.5
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
  # Taken together: the load lock is held (by the test) and two loaders wait on it, one asking `any` each. A loader that held
  # the card it had got while it waited would leave an `any` gate that is no load with no card — 75 after the card bound
  # (4 s here); with neither held, that gate takes a card at once. Each loader's seconds are its load-lock wait alone.
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  gate "$t/w1" any "$t/lease" BLOOMERY_GATE_V41_LOAD=1 &
  p1=$!
  gate "$t/w2" any "$t/lease" BLOOMERY_GATE_V41_LOAD=1 &
  p2=$!
  sleep 1
  rc=0 t0=$SECONDS
  gate "$t/out" any "$t/lease" || rc=$?
  el=$((SECONDS - t0))
  echo "elapsed ${el} s" >> "$t/out"
  judge 'together: two loaders queued on the load lock hold no card, an any gate takes one at once' "$rc" 0 'elapsed [0-2] s' 'no gate lock'
  judge '  … and takes it, not a queue' "$rc" 0 'ok on the (3090|a6000) \(asked any\)'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  # Each loader counts its own load-lock wait. A card wait may show too: at the instant the lock frees both ask for the one card
  # `any` picks, and the loser waits for it honestly, so no line here says a card was not waited for.
  cp "$t/w1" "$t/out"
  judge '  … the first queued loader runs once the lock frees, counting its load-lock wait' "$rc" 0 'waited [0-9]+ s for the V4\.1 load lock'
  cp "$t/w2" "$t/out"
  judge '  … and the second' "$rc" 0 'waited [0-9]+ s for the V4\.1 load lock'
  # The same for a loader of both cards: it took both card locks, then waited for the load lock holding them.
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  gate "$t/w1" both "$t/lease" BLOOMERY_BOX_CARD=both BLOOMERY_GATE_V41_LOAD=1 &
  p1=$!
  sleep 1
  rc=0 t0=$SECONDS
  gate "$t/out" any "$t/lease" || rc=$?
  el=$((SECONDS - t0))
  echo "elapsed ${el} s" >> "$t/out"
  judge 'together: a two-card loader queued on the load lock holds no card, an any gate takes one at once' "$rc" 0 'elapsed [0-2] s' 'no gate lock'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  cp "$t/w1" "$t/out"
  judge '  … and the loader runs on both cards once the lock frees' "$rc" 0 'on both cards, both gate locks \(asked both\), V4\.1 load lock'
  # The tier. A fixture load takes no V4.1 load lock — only when its model is the file box.sh resolved: the lock file is not opened
  # (a directory stands there, which a real load refuses with 69), a held lock is not waited for, a request without the resolved
  # file is refused nothing and takes the lock as a real load does.
  rm -f "$t/v41-load.lock"
  mkdir "$t/v41-load.lock"
  FX='BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_MODEL=/m/fx-00001-of-00001.gguf'
  case_ 'a real load whose lock file cannot be opened: 69, named' 69 'cannot open .*v41-load\.lock' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  # shellcheck disable=SC2086 # FX is K=V words
  case_ 'a fixture load opens no load lock and takes none' 0 'ok on the 3090 \(asked 3090\), tier fixture: no V4\.1 load lock' 3090 "$t/lease" \
    BLOOMERY_GATE_V41_LOAD=1 $FX BLOOMERY_REF_MODEL=/m/fx-00001-of-00001.gguf
  rmdir "$t/v41-load.lock"
  : > "$t/v41-load.lock"
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  rc=0 t0=$SECONDS
  # shellcheck disable=SC2086
  gate "$t/out" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 $FX BLOOMERY_REF_MODEL=/m/fx-00001-of-00001.gguf || rc=$?
  el=$((SECONDS - t0))
  echo "elapsed ${el} s" >> "$t/out"
  judge 'a fixture load does not wait for a held load lock' "$rc" 0 'elapsed [0-2] s' 'waits for the V4\.1'
  # shellcheck disable=SC2086
  case_ "a fixture tier with another model than the resolved fixture takes the lock (a held one: 75), and says so" 75 \
    "BLOOMERY_TIER=fixture, but BLOOMERY_REF_MODEL='/m/real\.gguf' is not the fixture box.sh resolved" \
    3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 $FX BLOOMERY_REF_MODEL=/m/real.gguf
  case_ 'a fixture tier with no resolved fixture takes the lock too' 75 'BLOOMERY_FIXTURE_MODEL=..\): the V4\.1 load lock is taken' \
    3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_TIER=fixture BLOOMERY_REF_MODEL=/m/real.gguf
  case_ 'a fixture tier that names no model at all takes the lock too' 75 "BLOOMERY_REF_MODEL='' is not the fixture box.sh resolved \\(BLOOMERY_FIXTURE_MODEL=''\\)" \
    3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_TIER=fixture
  flock -u 6
  exec 6>&-
  case_ 'a real tier names no fixture: the lock is taken as before' 0 'ok on the 3090 \(asked 3090\), V4\.1 load lock$' 3090 "$t/lease" \
    BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_TIER=real
  case_ 'BLOOMERY_TIER other than real or fixture: 64, named' 64 'BLOOMERY_TIER is real or fixture .*got .tier2.' 3090 "$t/lease" BLOOMERY_TIER=tier2
  # The bound: a lock that never frees is 75, named, after V41_BOUND (6 s under --test-locks).
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  case_ 'the V4.1 load lock never frees: 75 after the bound, named' 75 'V4\.1 load lock .* was not free within 6 s' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  flock -u 6
  exec 6>&-
  # The batch hold (the header). hold <owner> [age s] writes the file as the batch's writer leaves it, its mtime <age> seconds back
  # (python3's utime: BSD and GNU touch share no relative form); hold_age is the mtime's age; hrun <verb> <owner> runs the writer and
  # appends the file as it stands after (all three take a third argument, a directory: the hold of another --test-locks dir); judge_all <name> <rc> <want rc> <ERE>…: every ERE must match a line of the output, and one
  # that starts with ! must match none.
  hold() {
    printf 'owner=%s since=%s\n' "$1" "$(($(date +%s) - ${2:-0}))" > "${3:-$t}/batch.gpuhold"
    python3 -c 'import os, sys, time; a = time.time() - float(sys.argv[2]); os.utime(sys.argv[1], (a, a))' "${3:-$t}/batch.gpuhold" "${2:-0}"
  }
  hold_age() { python3 -c 'import os, sys, time; print(int(time.time() - os.stat(sys.argv[1]).st_mtime))' "$1"; }
  hrun() {
    rc=0
    bash "$self" --test-locks "$t" --gpuhold "$@" > "$t/out" 2>&1 || rc=$?
    if [ -e "$t/batch.gpuhold" ]; then
      printf 'after: %s age=%s\n' "$(head -1 "$t/batch.gpuhold")" "$(hold_age "$t/batch.gpuhold")" >> "$t/out"
    else
      echo 'after: no hold' >> "$t/out"
    fi
  }
  judge_all() {
    local name=$1 got=$2 want=$3 pat miss=
    shift 3
    n=$((n + 1))
    out=$(cat "$t/out")
    for pat in "$@"; do
      case $pat in
        '!'*) ! printf '%s\n' "$out" | grep -Eq -- "${pat#!}" || miss="$miss /$pat/" ;;
        *) printf '%s\n' "$out" | grep -Eq -- "$pat" || miss="$miss /$pat/" ;;
      esac
    done
    if [ "$got" = "$want" ] && [ -z "$miss" ]; then
      echo "ok $name"
    else
      bad=$((bad + 1))
      echo "FAIL $name: rc $got (want $want), failed:${miss:- none}"
      printf '%s\n' "$out" | sed 's/^/    | /'
    fi
  }
  rm -f "$t/batch.gpuhold"
  rc=0
  gate "$t/out" 3090 "$t/lease" || rc=$?
  judge_all 'no hold: the run goes ahead and names no hold' "$rc" 0 'ok on the 3090 \(asked 3090\)' '!batch-hold|batch hold'
  # A fresh hold of another owner, four runs at once: no owner, a prefix of the owner (not the owner), a V4.1 loader whose load lock is
  # held (the hold is read before the load lock's probe: the bound named is the hold's 4 s, not the load lock's 6 s), and a run of both
  # cards.
  hold other 0
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  gate "$t/h1" 3090 "$t/lease" &
  p1=$!
  gate "$t/h2" 3090 "$t/lease" BLOOMERY_BATCH_OWNER=othe &
  p2=$!
  gate "$t/h3" any "$t/lease" BLOOMERY_GATE_V41_LOAD=1 &
  p3=$!
  gate "$t/h4" both "$t/lease" BLOOMERY_BOX_CARD=both &
  p4=$!
  # Two more beside them, each in a directory of its own so each has its own hold: a hold that names no owner (up, never free), and a
  # hold released while its run waits.
  mkdir -p "$t/d2" "$t/d3"
  : > "$t/d2/batch.gpuhold"
  TL="$t/d2" gate "$t/h5" 3090 "$t/lease" &
  p5=$!
  hold other 0 "$t/d3"
  TL="$t/d3" gate "$t/h6" 3090 "$t/lease" &
  p6=$!
  # And a stale hold over a 3090 lock that stays taken: the run polls for the card for the whole bound and names the stale hold once.
  mkdir -p "$t/d4"
  hold other 400 "$t/d4"
  exec 5> "$t/d4/gate.lock"
  flock -x 5 || { echo "FAIL: the test cannot hold the d4 gate lock"; return 1; }
  TL="$t/d4" gate "$t/h7" 3090 "$t/lease" &
  p7=$!
  # d3's hold is released once its run is waiting on it (its first [batch-hold] line) and has polled once more — an event of the run's
  # own clock, not a fixed time after the launch: the others wait out their 4 s bound beside it.
  k=0
  while ! grep -q 'batch-hold' "$t/h6" 2> /dev/null && [ "$k" -lt 50 ]; do
    sleep 0.2
    k=$((k + 1))
  done
  sleep 1
  rm -f "$t/d3/batch.gpuhold"
  sleep 0.5
  # While they wait they hold no card lock: both are free for a try of their own.
  prc3=0 prca=0
  flock -n "$t/gate.lock" true || prc3=$?
  flock -n "$t/gate-a6000.lock" true || prca=$?
  echo "probe 3090 lock rc=$prc3 a6000 lock rc=$prca" > "$t/out"
  judge_all 'runs waiting on the hold hold no card lock' 0 0 'probe 3090 lock rc=0 a6000 lock rc=0$'
  rc=0
  wait "$p1" || rc=$?
  cp "$t/h1" "$t/out"
  judge_all 'a fresh hold of another owner: the run waits, 75 at the bound, a [batch-hold] line at once naming owner and age' "$rc" 75 \
    '\[batch-hold\] .* waits: a landing batch holds the GPUs \(.*batch\.gpuhold, owner other, up [0-9]+ s, refreshed [0-9]+ s ago\)' \
    'the batch hold .*batch\.gpuhold \(owner other\) was still up after [4-9] s .*contention' '!ok on the'
  rc=0
  wait "$p2" || rc=$?
  cp "$t/h2" "$t/out"
  judge_all '  … an owner that is a prefix of the hold'"'"'s is not the owner' "$rc" 75 '\[batch-hold\] .* owner other' '!ok on the'
  rc=0
  wait "$p3" || rc=$?
  cp "$t/h3" "$t/out"
  judge_all '  … a V4.1 loader waits on the hold before its load lock: the hold'"'"'s bound, not the load lock'"'"'s' "$rc" 75 \
    'the batch hold .* was still up after [4-9] s' '!V4\.1 load lock .* was not free'
  rc=0
  wait "$p4" || rc=$?
  cp "$t/h4" "$t/out"
  judge_all '  … a run of both cards waits too' "$rc" 75 '\[batch-hold\] .* owner other' '!ok on'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p5" || rc=$?
  cp "$t/h5" "$t/out"
  judge_all 'a hold that names no owner is up, never free: 75 at the bound, owner ?' "$rc" 75 '\[batch-hold\] .* owner \?, ' '!ok on'
  rc=0
  wait "$p6" || rc=$?
  cp "$t/h6" "$t/out"
  judge_all 'a hold released while the run waits: it goes ahead and counts the wait on its own line' "$rc" 0 \
    'ok on the 3090 \(asked 3090\)' '^gpu-gate\.sh: waited [0-9]+ s for the batch hold$'
  rc=0
  wait "$p7" || rc=$?
  flock -u 5
  exec 5>&-
  cp "$t/h7" "$t/out"
  echo "hold lines: $(grep -c 'batch-hold' "$t/out")" >> "$t/out"
  judge_all 'a stale hold over a taken card lock: the run polls on for the card, 75, and names the stale hold once' "$rc" 75 \
    'no gate lock \(3090\) was free within 4 s' '\[batch-hold\] .* is stale' 'hold lines: 1$'
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_BATCH_OWNER=other || rc=$?
  judge_all "the hold's own batch passes: its owner runs, no hold line" "$rc" 0 'ok on the 3090 \(asked 3090\)' '!batch-hold|batch hold'
  hold other 400
  rc=0
  gate "$t/out" 3090 "$t/lease" || rc=$?
  echo "hold lines: $(grep -c 'batch-hold' "$t/out")" >> "$t/out"
  judge_all 'a stale hold (not refreshed for 400 s): the run goes ahead, and says so once, naming the hold and its age' "$rc" 0 \
    'ok on the 3090 \(asked 3090\)' '!waits' \
    '\[batch-hold\] .*batch\.gpuhold \(owner other\) is stale — not refreshed for 4[0-9][0-9] s \(stale at 300 s\)' 'hold lines: 1$'
  case_ 'BLOOMERY_BATCH_OWNER with a character other than letters, digits, _: 64, named' 64 "BLOOMERY_BATCH_OWNER is the batch hold's owner .*got 'a-b'" 3090 "$t/lease" BLOOMERY_BATCH_OWNER=a-b
  # The writer: up, beat and down on the file, owner-checked; the wait above reads what it writes.
  hrun up batcha
  judge_all 'gpuhold up: puts the hold up with its owner and the box clock' "$rc" 0 'up: owner batcha$' 'after: owner=batcha since=[0-9]+ age=[0-2]$'
  hrun up batcha
  judge_all '  … again by the same owner: refreshed, not refused' "$rc" 0 'up: owner batcha$' 'after: owner=batcha since=[0-9]+ age=[0-2]$'
  hrun up other
  judge_all "  … another owner's fresh hold is refused, 75, and stays" "$rc" 75 'is up for owner batcha .*not taking it \(rc 75\)' 'after: owner=batcha ' '!owner=other'
  hold batcha 200
  hrun beat batcha
  judge_all 'gpuhold beat: refreshes the mtime of its own hold' "$rc" 0 'after: owner=batcha since=[0-9]+ age=[0-2]$'
  hold batcha 200
  hrun beat other
  judge_all "  … never another owner's" "$rc" 3 "owner batcha's, not other's: not refreshing it \(rc 3\)" 'after: owner=batcha since=[0-9]+ age=(199|20[0-9])$'
  rm -f "$t/batch.gpuhold"
  hrun beat batcha
  judge_all '  … and never creates a hold that is gone (a beat in flight after a down)' "$rc" 3 'is gone, not batcha.s: not refreshing it \(rc 3\)' 'after: no hold$'
  hold batcha 0
  hrun down other
  judge_all "gpuhold down: leaves another owner's hold" "$rc" 0 "owner batcha's, not other's: left up" 'after: owner=batcha ' '!down: owner'
  hrun down batcha
  judge_all '  … removes its own' "$rc" 0 'down: owner batcha$' 'after: no hold$'
  hrun down batcha
  judge_all '  … and is no error when it is already down' "$rc" 0 'is already down$' 'after: no hold$'
  hold other 400
  hrun up batcha
  judge_all 'gpuhold up: takes over a stale hold, naming it' "$rc" 0 'owner other was not refreshed for 4[0-9][0-9] s: taking it over' 'after: owner=batcha since=[0-9]+ age=[0-2]$'
  rm -f "$t/batch.gpuhold"
  hrun up 'a b'
  judge_all 'gpuhold: an owner with a space: 64, named' "$rc" 64 "an owner is letters, digits and _, got 'a b'" 'after: no hold$'
  # The file's name stays apart from the sittings' holds (/root/bloomery-<owner>-hold): builds and every other box command ignore it.
  out=$(bash "$self" --gpuhold path)
  out="$out $(env -u BLOOMERY_LEASE_HOLDS bash -c '. "$1"; case /root/bloomery-03-hold in $LEASE_HOLDS) printf "sitting-hold-matches " ;; esac; case "$2" in $LEASE_HOLDS) echo MATCHES ;; *) echo apart ;; esac' _ \
    "$(dirname "$self")/ref/lease-probe.sh" "$out")"
  printf '%s\n' "$out" > "$t/out"
  judge_all "the batch hold's name does not match the sittings' LEASE_HOLDS" 0 0 '^/root/bloomery-batch\.gpuhold sitting-hold-matches apart$'
  echo "gpu-gate self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi
if [ "${1:-}" = --test-locks ]; then
  [ $# -ge 3 ] && [ -d "$2" ] || { echo "usage: gpu-gate.sh --test-locks <dir> <binary> [args...] (the self-test's)" >&2; exit 64; }
  GATE_LOCK=$2/gate.lock A6000_LOCK=$2/gate-a6000.lock V41_LOCK=$2/v41-load.lock GPU_HOLD=$2/batch.gpuhold V41_BOUND=6 POLL=1 CARD_BOUND=4
  shift 2
fi
if [ "${1:-}" = --gpuhold ]; then
  shift
  gpuhold "$@"
  exit $?
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
# The tier (the header): the V4.1 load lock is for a real load. A fixture run takes it only when its model is not the
# resolved fixture file.
TIER=${BLOOMERY_TIER:-real}
case $TIER in
  real | fixture) ;;
  *) echo "gpu-gate.sh: BLOOMERY_TIER is real or fixture (unset: real), got '$TIER'" >&2; exit 64 ;;
esac
# BLOOMERY_BATCH_OWNER: the owner of the batch hold this run's batch put up (the header); the batch exports it into each item's box env.
case ${BLOOMERY_BATCH_OWNER:-} in
  *[!A-Za-z0-9_]*) echo "gpu-gate.sh: BLOOMERY_BATCH_OWNER is the batch hold's owner (letters, digits and _) or unset, got '$BLOOMERY_BATCH_OWNER'" >&2; exit 64 ;;
esac
TIER_NOTE=
if [ "$V41" = 1 ] && [ "$TIER" = fixture ]; then
  if [ -n "${BLOOMERY_FIXTURE_MODEL:-}" ] && [ "${BLOOMERY_REF_MODEL:-}" = "$BLOOMERY_FIXTURE_MODEL" ]; then
    V41=0
    TIER_NOTE=', tier fixture: no V4.1 load lock'
  else
    echo "gpu-gate.sh: BLOOMERY_TIER=fixture, but BLOOMERY_REF_MODEL='${BLOOMERY_REF_MODEL:-}' is not the fixture box.sh resolved (BLOOMERY_FIXTURE_MODEL='${BLOOMERY_FIXTURE_MODEL:-}'): the V4.1 load lock is taken, as for a real load" >&2
  fi
fi
# BLOOMERY_GATE_STACKS=<seconds>: the binary runs under tools/ref/stack-watch.sh, which dumps its thread
# stacks and ends it once its output has stopped for that long, so a hang says where before the bound.
STACKS=${BLOOMERY_GATE_STACKS:-}
case $STACKS in
  '') ;;
  0 | *[!0-9]*) echo "gpu-gate.sh: BLOOMERY_GATE_STACKS is whole seconds from 1 (the quiet time before a stack dump) or unset, got '$STACKS'" >&2; exit 64 ;;
esac
# BLOOMERY_GATE_GDB=1: the binary runs under gdb. A signal that ends it (a segfault in a driver's teardown, say)
# prints every thread's stack, and the runner exits 128 + the signal, as the bare run would; a clean exit keeps
# its code. A crash then says where in the run that printed it, with no core file to find.
GDB=${BLOOMERY_GATE_GDB:-0}
case $GDB in
  0 | 1) ;;
  *) echo "gpu-gate.sh: BLOOMERY_GATE_GDB is 1 (run under gdb) or 0/unset, got '$GDB'" >&2; exit 64 ;;
esac
if [ "$GDB" = 1 ] && [ -n "$STACKS" ]; then
  echo "gpu-gate.sh: BLOOMERY_GATE_GDB and BLOOMERY_GATE_STACKS each own the process; set one" >&2
  exit 64
fi
# 카드 고르기. 락은 카드마다 하나: 3090은 예전 경로 그대로(돌고 있는 트랙의 옛 사본이 그 경로를 잡는다),
# A6000은 새 파일. BLOOMERY_GATE_CARD=3090(기본 — 예전과 같다) | a6000 | any. any는 A6000을 먼저 본다 —
# V4.1 `--place gate` 게이트는 3090에서만 돌 수 있으니 떠도는 게이트가 3090을 비워 두는 편이 낫다. any는
# 잡힌 타이밍 임대가 재는 카드(아래 timing_card)를 건너뛰고, A6000은 컴퓨트 프로세스가 있어도 건너뛴다. 카드를
# 강제한 실행은 그 카드가 잡힌 임대의 타이밍 카드면 75로 끝난다. 임대 탐침은
# lease-probe.sh의 lease_free(공유 잠금) 하나다 — 대기 중 5초마다 도는 이 탐침이 배타 잠금이면 다른 탐침과 부딪혀
# 빈 임대를 잡힌 것으로 읽고, 테스트할 수 없는 임대도 비었다고 읽지 않는다.
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
LOCK_FILES=("9:$GATE_LOCK" "8:$A6000_LOCK")
[ "$V41" = 0 ] || LOCK_FILES+=("7:$V41_LOCK")
for lock in "${LOCK_FILES[@]}"; do
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
# take_card: one try, without blocking, at the card lock(s) this run asked for; GOT names what it holds, empty when none.
take_card() {
  GOT=
  case "$CARD" in
    3090) take_3090 || true ;;
    a6000) take_a6000 || true ;;
    any) take_a6000 || take_3090 || true ;;
  esac
}
# take_both: the 3090 lock first, then, holding it, the A6000's. Every other run holds one card lock and never waits while
# holding it, so this order cannot deadlock; a blocked flock wakes at the release before the polls of one-card runs
# (a poll that takes only when both are free starves while gates keep coming). Blocks up to what is left of CARD_BOUND.
take_both() {
  local left=$((CARD_BOUND - CARD_W)) start=$SECONDS
  [ "$left" -ge 1 ] || left=1
  GOT=
  if flock -w "$left" 9; then
    left=$((left - (SECONDS - start)))
    if flock -w "$((left > 0 ? left : 1))" 8; then GOT=both; else flock -u 9; fi
  fi
}
drop_card() {
  case "$GOT" in
    3090) flock -u 9 ;;
    a6000) flock -u 8 ;;
    both) flock -u 9; flock -u 8 ;;
  esac
  GOT=
}
# The V4.1 load lock: one V4.1 load on the box at a time, across trees and batches — two at once push each other's
# host set (~190 GB of the 256 GB) out of the page cache and both turn IO-bound. It is held by the binary for the whole
# run: the descriptor stays open, so the lock lives exactly as long as the load's process.
# The run takes it TOGETHER with its card lock(s): each poll tries the lock and then the card without blocking, and a run that
# got a card but not the lock lets the card go before it sleeps. A loader queued behind another load therefore holds no card
# while it waits (it held one for the other load's whole run, and an `any` gate found both cards taken with one idle); the
# poll's first look is a shared test of the lock, so while it is held the poll touches no card and no nvidia-smi.
# The only wait that holds a lock is take_both's second card lock, in the order above. The waits are the bounds' own:
# CARD_BOUND for a poll that found no card, V41_BOUND for one that found the card and not the lock, each 75 (contention, not a
# red gate) with the lock named; the load lock's holders are named at once and once a minute.
# Each poll's seconds go to one of the two waits, so the two `waited` lines never overlap: tools/gate-batch.sh subtracts both.
# The batch hold (the header) is read first in every poll, before the load lock's shared probe and before any card lock: a run under
# a hold holds nothing while it waits, and a gate already polling cannot slip in between a batch's items.
GOT=
CARD_W=0 LOAD_W=0 HOLD_W=0 why='' said=-60 hsaid=-60 last=$SECONDS
while :; do
  why=
  if batch_hold_blocks; then
    why=hold
  elif [ "$V41" = 1 ] && ! flock -s -n "$V41_LOCK" true 2> /dev/null; then
    why=load
  else
    if [ "$CARD" = both ]; then take_both; else take_card; fi
    if [ -z "$GOT" ]; then
      why=card
    elif [ "$V41" = 1 ] && ! flock -n 7; then
      drop_card
      why=load
    fi
  fi
  [ -n "$why" ] || break
  if [ "$why" = hold ] && [ $((HOLD_W - hsaid)) -ge 60 ]; then
    up_s='?'
    [ -z "$HOLD_SINCE" ] || up_s=$(($(date +%s) - HOLD_SINCE))
    echo "[batch-hold] $(now) $NAME waits: a landing batch holds the GPUs ($GPU_HOLD, owner $HOLD_OWNER, up $up_s s, refreshed $HOLD_AGE s ago), holding no lock; ${HOLD_W} s so far, at most ${CARD_BOUND} s with the lock waits" >&2
    hsaid=$HOLD_W
  fi
  if [ "$why" = load ] && [ $((LOAD_W - said)) -ge 60 ]; then
    echo "gpu-gate.sh: $NAME waits for the V4.1 load lock $V41_LOCK (another run is loading V4.1), holding no gate lock; ${LOAD_W} s so far, its holders:" >&2
    lease_holders "$V41_LOCK" >&2
    said=$LOAD_W
  fi
  sleep "$POLL"
  spent=$((SECONDS - last)) last=$SECONDS
  case $why in
    card) CARD_W=$((CARD_W + spent)) ;;
    hold) HOLD_W=$((HOLD_W + spent)) ;;
    *) LOAD_W=$((LOAD_W + spent)) ;;
  esac
  # The hold's seconds and the card lock's share CARD_BOUND: a batch's hold is a wait for the cards, as the card lock is.
  if [ $((CARD_W + HOLD_W)) -ge "$CARD_BOUND" ] && [ "$why" = card ]; then
    echo "gpu-gate.sh: no gate lock ($CARD) was free within ${CARD_BOUND} s — contention, not a red gate" >&2
    exit 75
  fi
  if [ $((CARD_W + HOLD_W)) -ge "$CARD_BOUND" ] && [ "$why" = hold ]; then
    echo "gpu-gate.sh: $NAME: the batch hold $GPU_HOLD (owner $HOLD_OWNER) was still up after ${HOLD_W} s (CARD_BOUND ${CARD_BOUND} s) — contention, not a red gate (rc 75)" >&2
    exit 75
  fi
  if [ "$LOAD_W" -ge "$V41_BOUND" ] && [ "$why" = load ]; then
    echo "gpu-gate.sh: $NAME: the V4.1 load lock $V41_LOCK was not free within ${V41_BOUND} s — contention, not a red gate (rc 75)" >&2
    exit 75
  fi
done
# What the last poll spent (take_both's blocking) was a wait for the card lock.
CARD_W=$((CARD_W + SECONDS - last))
# The waits are not the gate's time: tools/gate-batch.sh subtracts these lines' seconds from the item's times row.
[ "$HOLD_W" = 0 ] || echo "gpu-gate.sh: waited ${HOLD_W} s for the batch hold" >&2
[ "$CARD_W" = 0 ] || echo "gpu-gate.sh: waited ${CARD_W} s for the gate lock ($CARD)" >&2
[ "$LOAD_W" = 0 ] || echo "gpu-gate.sh: waited ${LOAD_W} s for the V4.1 load lock" >&2
if [ "$GOT" = a6000 ] || [ "$CARD" = any ]; then
  U=$(uuid_of "$([ "$GOT" = a6000 ] && echo A6000 || echo 3090)")
  [ -n "$U" ] || { echo "gpu-gate.sh: the $GOT lookup failed" >&2; exit 75; }
  export CUDA_VISIBLE_DEVICES=$U
fi
echo "gpu-gate.sh: $NAME on $([ "$GOT" = both ] && echo 'both cards, both gate locks' || echo "the $GOT") (asked $CARD)$([ "$V41" = 0 ] || echo ', V4.1 load lock')$TIER_NOTE" >&2
# The run's own wall, its locks' waits outside it, is the closing line: a recipe that runs several
# binaries, or one binary several times, is read per run.
RUN_T0=$SECONDS
if [ -n "$STACKS" ]; then
  bash "${BASH_SOURCE[0]%/*}/ref/stack-watch.sh" "$STACKS" "$NAME" -- timeout --kill-after=10 "$BOUND" "$EXE" "$@"
elif [ "$GDB" = 1 ]; then
  GDB_CMDS=$(mktemp)
  cat > "$GDB_CMDS" <<'G'
set pagination off
set print thread-events off
handle SIGPIPE nostop noprint
run
if $_isvoid($_exitcode)
  info threads
  thread apply all bt 30
  quit 128 + $_siginfo.si_signo
end
quit $_exitcode
G
  timeout --kill-after=10 "$BOUND" gdb -q -batch -x "$GDB_CMDS" --args "$EXE" "$@"
  rc=$?
  rm -f "$GDB_CMDS"
  (exit "$rc")
else
  timeout --kill-after=10 "$BOUND" "$EXE" "$@"
fi
rc=$?
echo "gpu-gate.sh: $NAME ran $((SECONDS - RUN_T0)) s (exit $rc)" >&2
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GPU GATE TIMED OUT: $NAME after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GPU GATE RED: $NAME (exit $rc)" >&2
fi
exit "$rc"
