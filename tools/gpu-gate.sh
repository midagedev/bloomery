#!/usr/bin/env bash
# GPU 게이트 러너 — 박스에서, box.sh가 들어간 원격 디렉터리에서 돈다. 레시피가 방금 지은
# target/release/<이름>을 카드의 게이트 락 아래에서 상한과 함께 돌리고, 그 종료 코드를
# 그대로 돌려준다. 이름 뒤의 인자는 바이너리에 그대로 붙는다.
#   ./tools/box.sh 'cargo oxide build … --bin gate_p3 && bash tools/gpu-gate.sh gate_p3'
# `--executable PATH [args...]` runs a compiled test executable with the same card/lock/bound
# contract. tools/gate.sh --gpu builds it before entering this runner and checks libtest's result.
#
# 막는 실패: 매달린 GPU 게이트 하나가 게이트 락을 무기한 쥐는 것. 락은 3090 게이트를 한 줄로 세우는 자리라,
# 한 게이트가 서면 그 락을 기다리는 트랙이 전부 선다. CPU 게이트의 900초 상한(tools/gate.sh)과 같은 값을
# 쓰고, 넘으면 빨간 게이트로 끝난다. 락과 상한의 소유자는 이 스크립트 하나다 — 레시피에 락 경로가 직접
# 나오면 check-recipes가 빨강이다.
#
# 종료 코드: 게이트의 것 그대로. 124 = 상한에서 TERM으로 끝남, 137 = TERM을 무시해 --kill-after의 KILL로
# 끝남, 75 = 30분 안에 카드 락이나 V4.1 적재 락을 못 잡음(카드 락 상한에는 배치 홀드를 기다린 시간이 든다), 또는 강제한 카드(3090·a6000)가 잡힌 타이밍 임대의
# 타이밍 카드임(모두 경쟁이지 게이트 실패가 아니다), 70 = 카드를 강제했는데 임대를 테스트할 수 없음, 64 = 사용법, 2 = 바이너리 없음,
# 69 = 게이트 락이나 V4.1 적재 락 파일을 열 수 없음(박스의 root 셸이 아님), 73 = 이 실행의 사설 디렉터리를 만들 수 없음(아래 단락).
# 사설 실행 디렉터리: 카고는 빌드마다 target/release/<이름>을 지우고 다시 하드링크한다(이미 신선한 바이너리도). 그 파일을 매핑해 도는
# 프로세스는 같은 이름을 빌드하는 다른 항목의 링크 갈이에 이미지를 잃는다(cuda-oxide의 앵커 로더가 /proc/self/map_files의 경로를 다시 열고,
# 지워진 이름은 "(deleted)"로 읽혀 열리지 않는다: `cannot read the binary containing the CUDA artifact anchor`). 게이트가 띄우는 형제
# 바이너리(generate_qwen3moe, bloomery-quantize, bloomery-serve …)도 같다 — 그 경로는 current_exe().with_file_name(..)이나
# current_exe().parent().parent().join(..)이라 게이트의 디렉터리에서 나온다. 그래서 모든 락을 쥔 뒤 — 기다리는 게이트는 링크를 쥐지 않는다 —
# 바이너리 디렉터리(--executable이 deps/ 안의 시험 산출물이면 그 위 디렉터리) 아래 `.run.<pid>/`를 만들고 그 디렉터리의 실행 파일 전부를
# 하드링크해 넣은 뒤(deps/ 산출물은 `.run.<pid>/deps/<이름>`) 그 안의 사본을 돌리고, 어떤 끝에서도(EXIT 트랩) 디렉터리째 지운다. 형제를
# 찾는 두 길이 사설 디렉터리 안에서 풀리므로 실행 도중의 빌드는 이 실행의 이미지를 건드리지 않는다. 링크를 못 만들면 `ln`의 말을 붙여
# 73으로 끝난다 — 공유 경로로 조용히 돌아가지 않는다(스냅샷 도중 이름이 사라지는 틈은 짧게 다시 시도한다). 죽은 실행이 남긴 `.run.<pid>`
# 디렉터리와 옛 `.<이름>.run.<pid>` 링크(pid가 없는 것)는 다음 실행이 지우고, 살아 있는 pid의 것은 건드리지 않는다.
# `--self-test`: 카드 고르기, 임대 거절, V4.1 적재 락, 사설 실행 디렉터리를 임시 디렉터리의 락·임대 파일과 가짜 nvidia-smi로 시험한다(리눅스,
# flock과 timeout이 필요하다 — 맥에는 없다). 시험만 쓰는 첫 인자 `--test-locks DIR`이 세 락과 배치 홀드 파일(batch.gpuhold), V4.1 적재 큐(v41-queue.d, v41-queue.seq)를 DIR 아래로 옮기고
# V4.1 적재 락의 상한을 6초로, 카드 락의 상한을 4초로, 폴링을 1초로 줄인다. GATE_TEST_CAP=<초> (the self-test's helper waiters only): both bounds become that safety cap; GATE_TEST_PACE=<FIFO prefix> (the self-test's paced loaders only): each poll rests until the test grants the next one.
# 환경: BLOOMERY_GATE_BOUND(초, 기본 900 — tools/gate.sh와 같은 레버), BLOOMERY_GATE_CARD(아래 카드 고르기),
# BLOOMERY_BOX_CARD(box.sh가 넘기는 카드 선택 — 아래), BLOOMERY_GATE_V41_LOAD(1이면 V4.1 적재 락도 잡는다 — 아래).
# BLOOMERY_BATCH_OWNER(letters, digits, _: the owner of the batch hold this run's batch put up — the batch hold paragraph below).
# BLOOMERY_BUILD_ONLY=1: the call ends 0 once the binary is found, before any lock, ticket or card (tools/gate-batch.sh's build lane: the recipe's build ran before this call); tools/gate.sh reads the same variable.
# The pack phase (tools/pack-for-driver.sh, which tools/gate.sh sources too): after the card pin check and before any lock, ticket or card (so
# no card is held while it packs), the executable is packed when it carries sm_12x PTX unpacked and the local driver (libcuda's
# cuDriverGetVersion) is older than CUDA 13.1, whose PTX compiler truncates a cc 12.x kernel's addresses and which bloomery-gpu refuses
# to hand PTX (docs/upstream/nvlabs-ledger.md #55). A script, a host binary, another target, a packed binary and a newer driver pass
# untouched; an already packed one (tools/oxart-pack.sh's 65) is no error. It runs under BLOOMERY_GATE_BUILD_BOUND (gate-bound.sh). A pack
# that fails ends the call with its own code after a `GATE RED (exit N) in the pack phase — no test ran` line (`GATE TIMED OUT in the
# pack phase` for the bound: 124 or 137), no binary run: 1 a cubin failed the address scan, 64 a refused bound, 69 a tool missing, 70 the
# executable's sections, SM12X_JIT_FIXED or the driver's version unreadable (70 too, the shell's own "No such file" line, when tools/pack-for-driver.sh
# itself is missing: the call ends before it reads its first argument). Only the red line tells those codes from the binary's own.
# BLOOMERY_GATE_STACKS(초 — 설정하면 바이너리의 출력이 그만큼 멈출 때 tools/ref/stack-watch.sh가 스레드 스택을 뜨고 끝낸다).
# BLOOMERY_LOCK_DIR: the directory of every lock and hold file named below, /root when unset (tools/ref/lease-probe.sh's LOCK_DIR).
# BLOOMERY_BOX_CARD=local (tools/box.sh's local mode): the run takes one lock, which holds every card this machine has, so it
# holds the cards in view whatever they are (tools/ref/cards.sh's cards_view), and leaves CUDA_VISIBLE_DEVICES as it finds it.
# BLOOMERY_GATE_CARD may be unset, 3090, a6000 or any, and both when two or more cards are in view (other names, and both with
# fewer: 64); the names are lane words, and each means the cards in view. A held timing lease is read as timing that card: the run
# waits as for its lock. Off the box the lock is bloomery-gate-local.lock under a lock directory that is never /root (64). On the
# box (/root/bloomery-cpu.lock exists, as lease_take reads it) the one lock is the gate lock the box's other runs take
# (/root/bloomery-gate.lock), the box's environment file declares the cards in view, and a BLOOMERY_LOCK_DIR other than /root is
# refused (64): the run would hold no lock the box's gates and sittings take.
# A card pin (CUDA_VISIBLE_DEVICES) that names no card nvidia-smi lists ends the run by name (64) before any lock is opened.
# BLOOMERY_CARD_HELD: exported as 1 to the binary once every lock is held (its card-byte invariant then refuses, never only records).
# BLOOMERY_GATE_GDB=1: the binary runs under gdb, which prints every thread's stack when a signal ends it.
# BLOOMERY_TIER=real|fixture (unset: real) and BLOOMERY_FIXTURE_MODEL: the fixture tier (tools/box.sh resolves both, see
# tools/ref/ref-paths.sh). A run asking for the V4.1 load lock takes none when its tier is fixture AND its
# BLOOMERY_REF_MODEL is the file box.sh resolved the tier to (BLOOMERY_FIXTURE_MODEL): the lock keeps two ~190 GB host sets
# out of each other's page cache, and a fixture's host set is a few GB. The request alone does not skip it: a fixture tier whose
# model is not the resolved fixture file (a caller's own BLOOMERY_REF_MODEL, a family the tier left on its real file) takes the
# lock as a real load does, and says so. A fixture run does not open the lock file at all. Any other BLOOMERY_TIER value is 64.
# Locks. A run takes its card lock(s) and, when it loads, the V4.1 load lock: the loaders are served in arrival order (the queue
# below), the head takes the load lock first and then its card(s), and every other run takes no card while it waits for a card or the
# lock, so an `any` gate takes a card a queued loader would have held. Nothing waits while holding a lock except the head (the load
# lock, for a card) and the two-card order (the 3090's lock, for the A6000's).
# The V4.1 load queue. A run that loads V4.1 (BLOOMERY_GATE_V41_LOAD=1 and a real tier; a fixture load takes no ticket, as it takes
# no lock) writes a ticket into /root/bloomery-v41-queue.d/ before its first poll and removes it once it holds the load lock and its
# card(s), and on every exit (a trap; a run killed outright leaves a ticket the next reader removes). The name is `<class>-<seq>-<pid>`:
# class from BLOOMERY_V41_CLASS (1 the lead's landing batch, 2 round gates, the default, 3 sittings, 4 idle fill; any other value is 64),
# seq ten digits from /root/bloomery-v41-queue.seq, read and incremented under a short flock, the ticket written inside it so that no
# lower seq is ever missing from the directory while a higher one is there. The content is one line, `owner=<BLOOMERY_BATCH_OWNER or the
# remote dir's basename> pid=<pid> start=<field 22 of /proc/<pid>/stat> since=<epoch> cards=<3090|a6000|both|any|local> item=<the binary and
# its args, up to ` tier=`> tier=real pred_s=<BLOOMERY_V41_PRED_S, the calling recipe's median wall in seconds, or ->`. The order key is
# (class, seq), ascending; the batch hold below stays above every class, and nothing preempts a holder. A ticket is live while the
# process's start time equals its start= (a reused pid does not keep one alive); any reader removes a dead ticket and says so once on
# stderr. Only the head live ticket tries a lock: a run behind it takes no card and no load lock, each poll it only re-reads the queue.
# A ticket a fresh batch hold holds back (its owner is not the hold's) is not counted: such a run waits under the hold and takes nothing,
# so as head it would stall the batch's own loaders, which pass the hold; and a head that holds the load lock when a hold goes up lets
# it go at its next poll and queues again at its place. `gpu-gate.sh --v41-queue` prints the live tickets in order and their age (a read: box.sh runs it
# under BLOOMERY_BOX_READONLY=1), removing a dead one it finds. Paths: V41_QDIR, V41_SEQ (--test-locks moves them with the locks).
# Lock order, checked path by path: the load lock (7) is taken before a card lock (3090: 9, A6000: 8), a card lock never before it, and the
# 3090's before the A6000's. Every blocking wait goes from a lower lock to a higher one, so none waits in a cycle:
#   - the head loader holds 7 and waits for a card: a one-card or `any` run by non-blocking tries each poll, `both` by take_both's
#     blocking flock on 9, then, holding 9, on 8. No holder of a card waits for 7: a loader takes 7 before any card; the one other order
#     is an older copy of this script on another track (card first, then a non-blocking `flock -n 7`, which drops the card at once).
#   - a non-loader gate, a one-card or `any` loader's polls (a try that wins one card stops there; `any` frees the A6000 it found busy before
#     it tries the 3090) and the fixture tier hold one card at most and never wait while holding it. take_both is the only run that waits
#     holding a card lock (9, for 8), and nothing waits for 7 while holding 9 or 8.
#   - tools/ref/dump-draft.sh and tools/ref/vision/dump-vision.sh wait on the 3090's lock (blocking, holding nothing) and take no load lock.
#   - the loader's binary inherits the three descriptors and takes no gate lock or load lock; BLOOMERY_HOST_LOCK locks pages, not a file.
#   - a non-head waiter and a run under the batch hold wait holding nothing. A head that polls for its card drops 7 at the next poll when a
#     hold goes up. A `both` head is inside take_both's blocking flock and reads the hold only after it, so the batch's own loaders can wait
#     behind it for up to CARD_BOUND: a cost, not a cycle (the head's wait ends when the cards are released).
# The batch hold. While the lead's landing batch (tools/gate-batch.sh --ledger) runs, the file /root/bloomery-batch.gpuhold is up on
# the box: `owner=<id> since=<epoch>`, its mtime refreshed by the batch's heartbeat. A run whose BLOOMERY_BATCH_OWNER (which the
# batch puts in each of its items' box env) is not that owner WAITS at the top of every poll — before the V4.1 load lock, before any
# card lock, so it holds none — printing `[batch-hold]` lines at once and once a minute, inside CARD_BOUND (75 at the bound, as
# contention is today, with its seconds in a `waited N s for the batch hold` line that tools/gate-batch.sh subtracts). The batch's own
# items pass. A hold whose mtime is 300 s old is stale (the batch or its link to the box is gone): the run ignores it, saying so once.
# A hold that cannot be read is read as up, never as free. A run with BLOOMERY_HOLD_YIELD=1 (the items of a round batch,
# tools/gate-batch.sh --round-ledger) does not wait: the first poll that finds a fresh hold of another owner up ends it with 75,
# naming the owner (`is up: yielding at the first probe`), holding no lock, so its lane takes its next item at once.
# The name does not match tools/ref/lease-probe.sh's LEASE_HOLDS
# (/root/bloomery-*-hold): builds and every non-GPU box command ignore it. The file has one writer, this script:
#   gpu-gate.sh --gpuhold up|beat|down <owner>   up: put it up (75 while another owner's fresh hold is up; a stale one is taken over);
#                                                beat: refresh its mtime, never create it (3: gone or another owner's);
#                                                down: remove it, only if it is <owner>'s; (owner: letters, digits, _)
#   gpu-gate.sh --gpuhold path                   print the path
# The batch calls these through tools/box.sh with BLOOMERY_BOX_READONLY=1 (no sync, no guard wait: a heartbeat stuck behind a sitting
# would go stale), a deliberate write of this one control file.
# The sitting hold. While a release sitting (tools/release/sitting.sh) runs, the file /root/bloomery-sitting.gpuhold is up on the box:
# `owner=<id> cards=<3090|a6000|3090,a6000|local> pid=<the sitting's> start=<its start time> since=<epoch>`. Its meaning is not the batch
# hold's, so it is a file of its own: nothing passes it (no gate of the sitting's owner exists), it names the cards the sitting times,
# and it lives as long as its pid does, with no heartbeat (the sitting runs on the box; a pid that is gone, or another process now,
# makes the hold dead and the runs ignore it, saying so once). The box hold /root/bloomery-<owner>-hold stops new box.sh commands; it
# cannot stop a gate whose box.sh command began before it, and that gate takes a card the moment the lease is free between two cells.
# A run whose cards meet the hold's WAITS at the top of every poll, after the batch hold and before the load lock and any card lock, so
# it holds none, printing `[sitting-hold]` lines at once and once a minute, inside CARD_BOUND (75 at the bound, with its seconds in a
# `waited N s for the sitting hold` line): a run of 3090 meets a hold on the 3090, a6000 one on the A6000, both and local any hold,
# and `any` only when both cards are held. `cards=local` is a sitting on a machine in local mode (BLOOMERY_BOX_CARD=local: the one lock holds
# every card the machine has, whatever it calls them), a hold on every card: no run goes by it (with one held it takes the other, and never the held one). A hold that cannot be read is
# read as up for both cards, never as free. The file has one writer, this script:
#   gpu-gate.sh --sitting-hold up <owner> <cards> <pid>   put it up (75 while a live hold is up, any owner's; a dead one is taken over)
#   gpu-gate.sh --sitting-hold down <owner>               remove it, only if it is <owner>'s
#   gpu-gate.sh --sitting-hold path                       print the path
# The name does not match tools/ref/lease-probe.sh's LEASE_HOLDS either: it is the sitting's second hold, the GPU one.
set -uo pipefail
# shellcheck source=tools/ref/lease-probe.sh
source "${BASH_SOURCE[0]%/*}/ref/lease-probe.sh"
GATE_LOCK=$LOCK_DIR/bloomery-gate.lock A6000_LOCK=$LOCK_DIR/bloomery-gate-a6000.lock LOCAL_LOCK=$LOCK_DIR/bloomery-gate-local.lock
V41_LOCK=$LOCK_DIR/bloomery-v41-load.lock V41_BOUND=1800 POLL=1 CARD_BOUND=1800
GPU_HOLD=$LOCK_DIR/bloomery-batch.gpuhold HOLD_STALE=300
SIT_HOLD=$LOCK_DIR/bloomery-sitting.gpuhold
V41_QDIR=$LOCK_DIR/bloomery-v41-queue.d V41_SEQ=$LOCK_DIR/bloomery-v41-queue.seq PACE=''
EXE_LINK=''

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

# The V4.1 load queue (the header). Q_NAME[i] is the ticket's file name, Q_LINE[i] its line and Q_OWNER[i] its owner, in key order.
Q_NAME=() Q_LINE=() Q_OWNER=() Q_N=0 Q_POS=0 Q_HEAD_ITEM='' Q_HEAD_OWNER='' Q_HEAD_SINCE='' TICKET='' TV='' PID_START=''

# pid_start <pid>: PID_START is the process's start time as a ticket's `start=` holds it (/proc/<pid>/stat field 22), empty when the
# process is gone or a zombie. Without /proc (the Mac, where the self-test runs) the `ps` start time stands in, its spaces dropped.
pid_start() {
  local line rest f
  PID_START=
  if [ -d /proc/self ]; then
    IFS= read -r line 2> /dev/null < "/proc/$1/stat" || return 0
    rest=${line##*) }
    read -r -a f <<< "$rest"
    [ "${f[0]:-Z}" != Z ] || return 0
    PID_START=${f[19]:-}
  else
    PID_START=$(ps -o lstart= -p "$1" 2> /dev/null | tr -d ' ') || PID_START=
  fi
}

# The sitting hold (the header). sit_hold_read: SIT_STATE none | up | dead (its pid is gone, or another process now), SIT_OWNER (`?`
# when the file names none), SIT_CARDS (3090, a6000, 3090,a6000 or local; a file that names none readable is read as both), SIT_PID, SIT_START
# and SIT_SINCE. A file whose pid cannot be read is up, never free.
SIT_STATE=none SIT_OWNER='' SIT_CARDS='' SIT_PID='' SIT_START='' SIT_SINCE='' SIT_DEAD_SAID=0
sit_hold_read() {
  local line f
  SIT_STATE=none SIT_OWNER='' SIT_CARDS='' SIT_PID='' SIT_START='' SIT_SINCE=''
  [ -e "$SIT_HOLD" ] || return 0
  SIT_STATE=up SIT_OWNER='?' SIT_CARDS='3090,a6000'
  line=$(head -1 "$SIT_HOLD" 2> /dev/null) || line=
  set -f
  for f in $line; do
    case $f in
      owner=*) SIT_OWNER=${f#owner=} ;;
      cards=*) SIT_CARDS=${f#cards=} ;;
      pid=*) SIT_PID=${f#pid=} ;;
      start=*) SIT_START=${f#start=} ;;
      since=*[!0-9]*) ;;
      since=?*) SIT_SINCE=${f#since=} ;;
    esac
  done
  set +f
  case $SIT_OWNER in '' | *[!A-Za-z0-9_]*) SIT_OWNER='?' ;; esac
  case $SIT_CARDS in 3090 | a6000 | 3090,a6000 | a6000,3090 | local) ;; *) SIT_CARDS='3090,a6000' ;; esac
  case $SIT_PID in '' | *[!0-9]*) return 0 ;; esac
  pid_start "$SIT_PID"
  if [ -z "$PID_START" ] || [ "$PID_START" != "$SIT_START" ]; then SIT_STATE=dead; fi
}

# sit_card_held <3090|a6000>: 0 when the hold (read by sit_hold_read) is up and names the card, or names `local`, which is every card.
sit_card_held() {
  [ "$SIT_STATE" = up ] || return 1
  case ",$SIT_CARDS," in *",$1,"* | *,local,*) return 0 ;; esac
  return 1
}

# sit_ticket_held <cards word of a queue ticket>: 0 when the hold (read by sit_hold_read) is up and stops a run of those cards.
sit_ticket_held() {
  [ "$SIT_STATE" = up ] || return 1
  case $1 in
    3090) sit_card_held 3090 ;;
    a6000) sit_card_held a6000 ;;
    any) sit_card_held 3090 && sit_card_held a6000 ;;
    *) return 0 ;;
  esac
}

# sit_hold_blocks: 0 when a live sitting hold meets this run's cards (the run waits), else 1. `any` waits only when both cards are
# held: with one held, take_3090 and take_a6000 leave that card alone. A dead hold is named once, then ignored.
sit_hold_blocks() {
  sit_hold_read
  case $SIT_STATE in
    up) ;;
    dead)
      if [ "$SIT_DEAD_SAID" = 0 ]; then
        echo "[sitting-hold] $(now) ${NAME:-gpu-gate.sh}: the sitting hold $SIT_HOLD (owner $SIT_OWNER, pid $SIT_PID) is dead — its sitting is gone; ignoring it" >&2
        SIT_DEAD_SAID=1
      fi
      return 1 ;;
    *) return 1 ;;
  esac
  case $CARD in
    3090) sit_card_held 3090 ;;
    a6000) sit_card_held a6000 ;;
    any) sit_card_held 3090 && sit_card_held a6000 ;;
    *) return 0 ;;
  esac
}

# sitting_hold <verb> ...: the sitting hold file's one writer (the header). Prints what it did; the rc is the verb's.
sitting_hold() {
  local verb=${1:-} owner=${2:-} cards=${3:-} pid=${4:-} tmp
  if [ "$verb" = path ]; then echo "$SIT_HOLD"; return 0; fi
  case $verb in
    up | down) ;;
    *) echo "gpu-gate.sh: --sitting-hold takes path, up <owner> <cards> <pid> or down <owner>, got '$verb'" >&2; return 64 ;;
  esac
  case $owner in
    '' | *[!A-Za-z0-9_]*) echo "gpu-gate.sh: --sitting-hold $verb: an owner is letters, digits and _, got '$owner'" >&2; return 64 ;;
  esac
  sit_hold_read
  if [ "$verb" = down ]; then
    if [ "$SIT_STATE" = none ]; then
      echo "gpu-gate.sh: the sitting hold $SIT_HOLD is already down"
    elif [ "$SIT_OWNER" != "$owner" ]; then
      echo "gpu-gate.sh: the sitting hold $SIT_HOLD is owner $SIT_OWNER's, not $owner's: left up"
    else
      rm -f "$SIT_HOLD"
      echo "gpu-gate.sh: sitting hold $SIT_HOLD down: owner $owner"
    fi
    return 0
  fi
  case $cards in
    3090 | a6000 | 3090,a6000 | local) ;;
    *) echo "gpu-gate.sh: --sitting-hold up: the cards are 3090, a6000, 3090,a6000 or local, got '$cards'" >&2; return 64 ;;
  esac
  case $pid in '' | *[!0-9]*) echo "gpu-gate.sh: --sitting-hold up: the pid is the sitting's, got '$pid'" >&2; return 64 ;; esac
  pid_start "$pid"
  [ -n "$PID_START" ] || { echo "gpu-gate.sh: --sitting-hold up: pid $pid is not a live process" >&2; return 64; }
  case $SIT_STATE in
    up)
      echo "gpu-gate.sh: the sitting hold $SIT_HOLD is up for owner $SIT_OWNER (pid $SIT_PID): not taking it (rc 75)" >&2
      return 75 ;;
    dead)
      echo "gpu-gate.sh: the sitting hold $SIT_HOLD of owner $SIT_OWNER (pid $SIT_PID) is dead: taking it over" >&2
      rm -f "$SIT_HOLD" ;;
  esac
  tmp=$SIT_HOLD.$$.new
  printf 'owner=%s cards=%s pid=%s start=%s since=%s\n' "$owner" "$cards" "$pid" "$PID_START" "$(date +%s)" > "$tmp" 2> /dev/null ||
    { echo "gpu-gate.sh: cannot write $tmp — the hold is written on the box, as root" >&2; return 69; }
  if ln "$tmp" "$SIT_HOLD" 2> /dev/null; then
    rm -f "$tmp"
  else
    rm -f "$tmp"
    echo "gpu-gate.sh: another sitting put $SIT_HOLD up at the same moment: not taking it (rc 75)" >&2
    return 75
  fi
  echo "gpu-gate.sh: sitting hold $SIT_HOLD up: owner $owner cards $cards"
}

# tget <ticket line> <key>: TV is the key's value (up to the next space), empty when the line has no such key.
tget() {
  local r=" $1"
  TV=
  case $r in *" $2="*) r=${r#*" $2="}; TV=${r%% *} ;; esac
}

# v41_queue_read: removes the dead tickets (once each, on stderr) and fills Q_NAME, Q_LINE and Q_OWNER with the live ones in key
# order. The class digit and the zero-padded seq make the C-locale glob order the key order.
v41_queue_read() {
  local LC_ALL=C f name line pid start why
  Q_NAME=() Q_LINE=() Q_OWNER=()
  for f in "$V41_QDIR"/[1-4]-*; do
    [ -f "$f" ] || continue
    name=${f##*/}
    line=
    IFS= read -r line 2> /dev/null < "$f" || [ -e "$f" ] || continue
    tget "$line" pid
    pid=$TV
    tget "$line" start
    start=$TV
    why=
    case $pid in '' | *[!0-9]*) why="no valid pid in it" ;; esac
    if [ -z "$why" ]; then
      pid_start "$pid"
      if [ -z "$PID_START" ]; then
        why="pid $pid is gone"
      elif [ "$PID_START" != "$start" ]; then
        why="pid $pid is another process now (start $PID_START, the ticket's $start)"
      fi
    fi
    if [ -n "$why" ]; then
      rm "$f" 2> /dev/null && echo "gpu-gate.sh: removed the dead V4.1 queue ticket $name ($why)" >&2
      continue
    fi
    tget "$line" owner
    Q_NAME+=("$name")
    Q_LINE+=("$line")
    Q_OWNER+=("${TV:-?}")
  done
}

# v41_standing <ticket name>: reads the queue (batch_hold_read and sit_hold_read have just run: HOLD_STATE, HOLD_OWNER, SIT_STATE, SIT_CARDS)
# and sets Q_N, the live tickets a fresh hold of another owner does not hold back and the sitting hold does not stop (a ticket whose cards
# meet the hold's), Q_POS, the named ticket's place among them (0: it is not there) and the head's item, owner and since. Returns 0 when
# the named ticket is the head.
v41_standing() {
  local i
  v41_queue_read
  Q_N=0 Q_POS=0 Q_HEAD_ITEM='' Q_HEAD_OWNER='' Q_HEAD_SINCE=''
  for ((i = 0; i < ${#Q_NAME[@]}; i++)); do
    if [ "$HOLD_STATE" = up ] && [ "${Q_OWNER[$i]}" != "$HOLD_OWNER" ]; then continue; fi
    tget "${Q_LINE[$i]}" cards
    if sit_ticket_held "$TV"; then continue; fi
    Q_N=$((Q_N + 1))
    if [ "$Q_N" = 1 ]; then
      Q_HEAD_OWNER=${Q_OWNER[$i]}
      tget "${Q_LINE[$i]}" since
      Q_HEAD_SINCE=$TV
      Q_HEAD_ITEM=${Q_LINE[$i]#*" item="}
      Q_HEAD_ITEM=${Q_HEAD_ITEM% tier=*}
    fi
    [ "${Q_NAME[$i]}" != "$1" ] || Q_POS=$Q_N
  done
  [ "$Q_POS" = 1 ]
}

# queued_say: the queue line, on arrival and once a minute.
queued_say() {
  local age='?'
  case $Q_HEAD_SINCE in '' | *[!0-9]*) ;; *) age=$(($(date +%s) - Q_HEAD_SINCE)) ;; esac
  echo "gpu-gate.sh: $NAME queued at $Q_POS of $Q_N for the V4.1 load lock (head: $Q_HEAD_ITEM $Q_HEAD_OWNER, $age s)" >&2
}

# v41_queue_view: `gpu-gate.sh --v41-queue`, the live tickets in order, one line each with the ticket's age.
v41_queue_view() {
  local i now age
  v41_queue_read
  now=$(date +%s)
  for ((i = 0; i < ${#Q_NAME[@]}; i++)); do
    tget "${Q_LINE[$i]}" since
    age='?'
    case $TV in '' | *[!0-9]*) ;; *) age=$((now - TV)) ;; esac
    echo "${Q_NAME[$i]} ${Q_LINE[$i]} age=$age"
  done
}

# ticket_write: this run's ticket, TICKET its path. The seq and the file are written inside one flock on the seq file, the file by a
# rename, so a reader sees a whole line and never a lower seq missing beside a higher one.
ticket_write() {
  local start owner item name
  pid_start "$$"
  start=$PID_START
  [ -n "$start" ] || { echo "gpu-gate.sh: cannot read this run's start time (pid $$) for its V4.1 queue ticket" >&2; exit 69; }
  owner=${BLOOMERY_BATCH_OWNER:-${PWD##*/}}
  owner=${owner//[^A-Za-z0-9._-]/_}
  item="$NAME${*:+ $*}"
  item=${item//[$'\n\r\t']/ }
  mkdir -p "$V41_QDIR" 2> /dev/null || { echo "gpu-gate.sh: cannot create $V41_QDIR — the gate runs on the box, as root" >&2; exit 69; }
  TICKET=$(
    {
      flock -x 10 || exit 1
      seq=
      IFS= read -r seq 2> /dev/null < "$V41_SEQ" || true
      if [[ $seq == *[!0-9]* ]]; then echo "gpu-gate.sh: $V41_SEQ holds '$seq', not a number" >&2; exit 1; fi
      seq=$((10#${seq:-0} + 1))
      printf '%s\n' "$seq" > "$V41_SEQ" || exit 1
      printf -v name '%s-%010d-%s' "$CLASS" "$seq" "$$"
      printf 'owner=%s pid=%s start=%s since=%s cards=%s item=%s tier=real pred_s=%s\n' "${owner:-?}" "$$" "$start" "$(date +%s)" "$CARD" "$item" "$PRED" \
        > "$V41_QDIR/.new.$$" && mv "$V41_QDIR/.new.$$" "$V41_QDIR/$name" || exit 1
      echo "$V41_QDIR/$name"
    } 10>> "$V41_SEQ"
  ) || { echo "gpu-gate.sh: cannot write a ticket under $V41_QDIR (above) — the gate runs on the box, as root" >&2; exit 69; }
}

# ticket_drop: removes this run's ticket.
ticket_drop() {
  [ -z "$TICKET" ] || rm -f "$TICKET"
  TICKET=
}

# The paced loader's `exit` line is written while the process still holds its locks (they go with the process, after the trap), so the
# test takes it as an end only once the pid is gone (ack). GATE_TEST_LINGER=<s> (the self-test's paced loaders only) widens that gap.
# on_exit: the EXIT trap of a run that holds something: a V4.1 loader's ticket, the run-private directory of the binaries (exe_link). In the
# self-test a paced loader (PACE) ends with one more line on its ack FIFO.
on_exit() {
  ticket_drop
  case $EXE_LINK in */.run.$$) rm -rf -- "$EXE_LINK" ;; esac
  EXE_LINK=
  [ -z "$PACE" ] || echo "exit" > "$PACE.ack"
  [ -z "$PACE" ] || [ -z "${GATE_TEST_LINGER:-}" ] || sleep "$GATE_TEST_LINGER"
}

# exe_link: after every lock is held, the run gets a private copy of its binary directory: <bin>/.run.<pid>/ holds a hard link to every
# executable file of <bin> (the gate's own directory, or the one above deps/ when EXE is a test artifact in deps/, which then sits at
# .run.<pid>/deps/<name>), and EXE becomes the link there. A build re-links target/release/<name> (removes it and links it again, fresh or
# not), which a process that mapped the shared name does not survive: the CUDA anchor loader re-opens its mapped image by path, and a
# removed name reads "(deleted)". The siblings a gate spawns are found from its own directory (current_exe().with_file_name(..), or
# parent().parent() for a deps/ artifact), so they run from the private copies too, and no build of the shared names reaches this run.
# The snapshot is taken under the locks in one `ln` call per try; a name that vanishes while a build re-links it fails the call, and the
# try is repeated (5 times, 0.1 s apart). A directory that cannot be made ends the run with 73, never a run from the shared path. First,
# the leftovers whose pid is gone (a run killed outright leaves its directory; the previous scheme left `.<NAME>.run.<pid>` links) are
# removed; a live pid's is left, and this run's own pid is no live owner. The EXIT trap (on_exit) removes the directory.
exe_link_snapshot() { # <bin dir> <EXE's dir> <run dir>
  local f files=()
  mkdir -p "$3" || return 1
  for f in "$1"/*; do
    [ -f "$f" ] && [ -x "$f" ] && files+=("$f")
  done
  if [ "${#files[@]}" -gt 0 ]; then ln -f "${files[@]}" "$3/" || return 1; fi
  [ "$1" = "$2" ] || { mkdir -p "$3/deps" && ln -f "$EXE" "$3/deps/"; }
}

exe_link() {
  local dir=${EXE%/*} name=${EXE##*/} bin d l p err try=0
  bin=$dir
  [ "${dir##*/}" != deps ] || bin=${dir%/*}
  for l in "$bin"/.run.* "$bin"/.*.run.*; do
    [ -e "$l" ] || [ -L "$l" ] || continue
    p=${l##*.run.}
    case $p in '' | *[!0-9]*) continue ;; esac
    [ "$p" != "$$" ] || continue
    if kill -0 "$p" 2> /dev/null || [ -d "/proc/$p" ]; then continue; fi
    rm -rf -- "$l"
    echo "gpu-gate.sh: $NAME: removed the stale run-private $l (pid $p is gone)" >&2
  done
  d=$bin/.run.$$
  rm -rf -- "$d"
  until err=$(exe_link_snapshot "$bin" "$dir" "$d" 2>&1); do
    try=$((try + 1))
    [ "$try" -lt 5 ] || {
      rm -rf -- "$d"
      echo "gpu-gate.sh: $NAME: cannot make its run-private directory $d from $bin (rc 73): $err — the gate does not run from the shared path" >&2
      exit 73
    }
    sleep 0.1
  done
  EXE_LINK=$d
  trap on_exit EXIT
  if [ "$bin" = "$dir" ]; then EXE=$d/$name; else EXE=$d/deps/$name; fi
  [ -x "$EXE" ] || {
    echo "gpu-gate.sh: $NAME: its run-private copy $EXE is missing after the snapshot (rc 73)" >&2
    exit 73
  }
}

# pace: what a poll rests with: POLL seconds of sleep. A paced loader (the self-test's, PACE=<FIFO prefix>) writes `polled` to
# <prefix>.ack and rests until the test writes a line to <prefix>.go, so the test, not the clock, decides when it polls next.
pace() {
  if [ -n "$PACE" ]; then
    echo polled > "$PACE.ack"
    read -r _ < "$PACE.go" || true
  else
    sleep "$POLL"
  fi
}

# The runner's own tests: each case runs this script with --test-locks <tmp> (the three locks under <tmp>,
# the V4.1 load lock's bound 6 s, the card locks' 4 s, the polls 1 s), BLOOMERY_LEASE_LOCK at a file the test holds or not, a stub
# nvidia-smi on PATH (two cards, no compute process) and a stub target/release/ok that, when asked (STUB_PID, STUB_TERM, STUB_READY,
# STUB_ORDER, STUB_HOLD), writes its pid and its parent's (the binary's timeout), exits STUB_TERM when it gets a TERM, says it
# is running, appends its STUB_ID to a marker file and holds until the test releases it. Concurrent runs are started with & and waited
# for by their $!. One line per case;
# exit 0 iff none failed.
# The cases wait on the runner's own bounds (the ones they judge: 4 s, 6 s) and on events (a line, a file, a pid that ends), so they
# run as SELFTEST_PARTS processes at once: each part is the cases between its `if gg_part K; then … fi` guards (contiguous in the file;
# the helpers and locals between them run in every part). `--self-test` starts every part and sums their counts; `--self-test --part K`
# runs one (K, or `all`: the parts one after another here).
# No verdict reads the machine's load: a wait that is not a judged bound ends on its event, and its clock (fs seconds for one wait, cap
# seconds for a helper's whole life) only catches a hang. A run that is late is waited for while its process lives.
# GPU_GATE_SELFTEST_LOAD=N (unset: off, and not set by CI): N spinning processes beside this process's cases. Each part
# process of `--self-test` does not start its own; the parent does. The pids are this shell's, stopped when it exits.
SELFTEST_PARTS=4
SELFTEST_LOAD_PIDS=()
selftest_load_up() {
  local n=${GPU_GATE_SELFTEST_LOAD:-} i
  SELFTEST_LOAD_PIDS=()
  [ -n "$n" ] || return 0
  case $n in
    *[!0-9]* | 0) echo "gpu-gate.sh: GPU_GATE_SELFTEST_LOAD is a whole number from 1, got '$n'" >&2; return 64 ;;
  esac
  for ((i = 0; i < n; i++)); do
    yes > /dev/null &
    SELFTEST_LOAD_PIDS+=("$!")
  done
}
selftest_load_down() {
  local p
  for p in ${SELFTEST_LOAD_PIDS[@]+"${SELFTEST_LOAD_PIDS[@]}"}; do
    kill "$p" 2> /dev/null || true
    wait "$p" 2> /dev/null || true
  done
  SELFTEST_LOAD_PIDS=()
}
self_test() {
  local self t n=0 bad=0 out rc rc1 p1 r1 r2 prc since fs=300 cap=870 part=${1:-all}
  gg_part() { [ "$part" = all ] || [ "$part" = "$1" ]; }
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  # shellcheck source=tools/ref/cancel-stubs.sh
  source "${self%/*}/ref/cancel-stubs.sh"
  # The cases pin the box's /root paths, so a lock directory from local mode is dropped; the local cases name their own.
  unset BLOOMERY_LOCK_DIR
  t=$(mktemp -d "${TMPDIR:-/tmp}/gpu-gate-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "selftest_load_down; rm -rf '$t'" EXIT
  selftest_load_up || return $?
  mkdir -p "$t/bin" "$t/tree/target/release"
  # The stub nvidia-smi is written before the tree's cards.sh is sourced (below, once PATH holds
  # it), so the resolution reads made-up UUIDs (depth-stub-cards.sh's, never the box's) and the
  # self-test needs no card: name,uuid is cards.sh's query, uuid,name uuid_of's further down.
  printf '%s\n' '#!/bin/sh' 'case "$*" in' \
    '  *query-gpu=name,uuid*) printf "NVIDIA GeForce RTX 3090, GPU-11111111-1111-1111-1111-111111111111\nNVIDIA RTX A6000, GPU-00000000-0000-0000-0000-000000000000\n" ;;' \
    '  *query-gpu=*) printf "GPU-11111111-1111-1111-1111-111111111111, NVIDIA GeForce RTX 3090\nGPU-00000000-0000-0000-0000-000000000000, NVIDIA RTX A6000\n" ;;' \
    '  *query-compute-apps=*) [ -z "${STUB_COMPUTE_UUID:-}" ] || printf "%s\n" "$STUB_COMPUTE_UUID"; exit "${STUB_COMPUTE_RC:-0}" ;;' \
    'esac' > "$t/bin/nvidia-smi"
  printf '%s\n' '#!/bin/sh' '[ -z "${STUB_PID:-}" ] || echo "$$ $PPID" > "$STUB_PID"' \
    '[ -z "${STUB_TERM:-}" ] || trap "exit $STUB_TERM" TERM' \
    '[ -z "${STUB_READY:-}" ] || echo up > "$STUB_READY"' '[ -z "${STUB_ORDER:-}" ] || echo "${STUB_ID:-?}" >> "$STUB_ORDER"' \
    '[ -z "${STUB_HOLD:-}" ] || read -r _ < "$STUB_HOLD"' \
    'echo "ran on $CUDA_VISIBLE_DEVICES"' > "$t/tree/target/release/ok"
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
  command -v timeout > /dev/null || cp "$(dirname "$self")/ref/stub-timeout.py" "$t/bin/timeout"
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
  local gate_args=()
  local gate_env=(env -u BLOOMERY_BOX_CARD -u BLOOMERY_CARD_HELD -u BLOOMERY_GATE_BOUND -u BLOOMERY_GATE_V41_LOAD -u BLOOMERY_GATE_STACKS -u BLOOMERY_GATE_GDB -u CUDA_VISIBLE_DEVICES
    -u BLOOMERY_TIER -u CARGO_BUILD_TARGET -u BLOOMERY_FIXTURE_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_BATCH_OWNER -u BLOOMERY_HOLD_YIELD -u BLOOMERY_GATE_BUILD_BOUND -u BLOOMERY_V41_CLASS -u BLOOMERY_V41_PRED_S
    -u GATE_TEST_PACE -u GATE_TIMEOUT_PID -u STUB_ORDER -u STUB_ID -u STUB_COMPUTE_UUID -u STUB_COMPUTE_RC PATH="$t/bin:$PATH")
  # gate <out file> <BLOOMERY_GATE_CARD> <lease file> [K=V…]: one run of the runner, its rc (TL=<dir>: the dir its locks and its
  # batch hold live in, default the test's own; GB=<s>: its own bound, default the failsafe $fs s, and for a run that takes
  # GATE_TEST_CAP (a helper waiter or a paced loader, which the test lets go by events) 30 s over its cap; GK=<s>: its kill-after,
  # default 5)
  gate() {
    local o=$1 c=$2 l=$3 gb=${GB:-$fs}
    shift 3
    case " $* " in *" GATE_TEST_CAP="*) gb=${GB:-$((cap + 30))} ;; esac
    # Under a failsafe bound: a runner that hangs ends the case red (124), it does not hang the check; a run that is only late ends itself.
    # GATE_TIMEOUT_PID, when set, is the timeout monitor: the background pid, the process a SIGALRM ends.
    # $$ in a subshell is the invoking shell, so the pid is $! of this timeout, not $$.
    (
      cd "$t/tree" || exit 70
      timeout --kill-after="${GK:-5}" "$gb" "${gate_env[@]}" \
        BLOOMERY_GATE_CARD="$c" BLOOMERY_LEASE_LOCK="$l" BLOOMERY_LEASE_PROC="$t/proc" "$@" \
        bash "$self" --test-locks "${TL:-$t}" "${GATE_BIN:-ok}" ${gate_args[@]+"${gate_args[@]}"} &
      tp=$!
      if [ -n "${GATE_TIMEOUT_PID:-}" ]; then
        printf '%s\n' "$tp" > "$GATE_TIMEOUT_PID" || { kill "$tp" 2> /dev/null || true; exit 70; }
      fi
      wait "$tp"
    ) > "$o" 2>&1
  }
  # judge <name> <rc> <want rc> <ERE that must match> [ERE that must not]. The output is matched as a here-string, never piped into
  # grep -q: under pipefail a writer still writing when grep exits on its first match fails the pipeline (Broken pipe), which read a
  # match as a miss.
  judge() {
    n=$((n + 1))
    out=$(cat "$t/out")
    if [ "$2" = "$3" ] && grep -Eq -- "$4" <<< "$out" && { [ -z "${5:-}" ] || ! grep -Eq -- "$5" <<< "$out"; }; then
      echo "ok $1"
    else
      bad=$((bad + 1))
      echo "FAIL $1: $(helper_cap)rc $2 (want $3), /$4/${5:+, not /$5/}"
      sed 's/^/    | /' <<< "$out"
    fi
  }
  # judge_all <name> <rc> <want rc> <ERE>…: every ERE must match a line of the output ($t/out), and one that starts with ! must match none.
  judge_all() {
    local name=$1 got=$2 want=$3 pat miss=
    shift 3
    n=$((n + 1))
    out=$(cat "$t/out")
    for pat in "$@"; do
      case $pat in
        '!'*) ! grep -Eq -- "${pat#!}" <<< "$out" || miss="$miss /$pat/" ;;
        *) grep -Eq -- "$pat" <<< "$out" || miss="$miss /$pat/" ;;
      esac
    done
    if [ "$got" = "$want" ] && [ -z "$miss" ]; then
      echo "ok $name"
    else
      bad=$((bad + 1))
      echo "FAIL $name: rc $got (want $want), failed:${miss:- none}"
      sed 's/^/    | /' <<< "$out"
    fi
  }
  # helper_cap: names a helper waiter that ended on its safety cap ($cap s, GATE_TEST_CAP) in a red's reason: it ends 75, never 0.
  helper_cap() { if grep -Eq "free within $cap s" <<< "$out"; then echo "helper waiter hit its cap ($cap s); "; fi; }
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
  if gg_part 1; then
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
  fi
  flock -u 7
  exec 7>&-
  if gg_part 1; then
  record 101 "$GPU_3090"
  case_ 'lease free: the record is not read, 3090 forced runs' 0 'ran on $' 3090 "$t/lease"
  rm -rf "$t/lease.card" "$t/proc"
  mkdir -p "$t/proc"
  case_ 'a6000 forced, lease free: the A6000' 0 "ran on $GPU_A6000" a6000 "$t/lease"
  # BLOOMERY_BUILD_ONLY=1 (the batch's build lane): the call ends 0 once the binary is found, before any lock, ticket or card — a held
  # lease with no record, which refuses a run, changes nothing. A missing binary is still named, any other value ends 64, and 0 runs.
  exec 7> "$t/lease"
  flock -x 7 || { echo "FAIL: the test cannot hold its lease file"; return 1; }
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_BUILD_ONLY=1 || rc=$?
  judge 'build-only: under a held lease the call ends 0 once the binary is found, and nothing runs' "$rc" 0 'ok is built; nothing runs, no lock is taken' 'ran on|in doubt'
  mkdir -p "$t/mut"
  for bf in "$(dirname "$self")"/*; do [ "${bf##*/}" = gpu-gate.sh ] || ln -sf "$bf" "$t/mut/${bf##*/}"; done
  sed '/^case \${BLOOMERY_BUILD_ONLY:-0} in$/,/^esac$/d' "$self" > "$t/mut/gpu-gate.sh"
  n=$((n + 1))
  if [ "$(grep -c 'BLOOMERY_BUILD_ONLY: \$EXE is built' "$t/mut/gpu-gate.sh")" = 0 ] && [ "$(grep -c 'BLOOMERY_BUILD_ONLY: \$EXE is built' "$self")" = 1 ]; then echo "ok build-only mutant: the short-circuit is removed from the copy"
  else bad=$((bad + 1)); echo "FAIL build-only mutant: the sed did not remove the short-circuit"; fi
  rc=0
  self=$t/mut/gpu-gate.sh gate "$t/out" 3090 "$t/lease" BLOOMERY_BUILD_ONLY=1 || rc=$?
  n=$((n + 1))
  out=$(cat "$t/out")
  if grep -Eq 'ran on|in doubt' <<< "$out"; then echo "ok build-only mutant: without the short-circuit the run reaches the lease (or the stub), rc $rc"
  else bad=$((bad + 1)); echo "FAIL build-only mutant: without the short-circuit the call reached neither the lease nor the stub (rc $rc)"; sed 's/^/    | /' <<< "$out"; fi
  flock -u 7
  exec 7>&-
  GATE_BIN=nosuch case_ 'build-only: a binary the recipe did not build is named first, rc 2' 2 'no executable file .*nosuch' 3090 "$t/lease" BLOOMERY_BUILD_ONLY=1
  case_ 'build-only: any other value ends 64 before a lock' 64 "BLOOMERY_BUILD_ONLY is 1 .* got '2'" 3090 "$t/lease" BLOOMERY_BUILD_ONLY=2
  case_ 'build-only: 0 runs the gate as ever' 0 'ran on' 3090 "$t/lease" BLOOMERY_BUILD_ONLY=0
  case_ 'any, successful empty process query: takes the A6000' 0 'ok on the a6000 \(asked any\)' any "$t/lease"
  case_ 'any, successful process query names the A6000: takes the 3090' 0 'ok on the 3090 \(asked any\)' any "$t/lease" STUB_COMPUTE_UUID="$GPU_A6000"
  case_ 'any, successful process query names another card: takes the A6000' 0 'ok on the a6000 \(asked any\)' any "$t/lease" STUB_COMPUTE_UUID="$GPU_3090"
  case_ 'any, failed empty process query: takes the 3090' 0 'ok on the 3090 \(asked any\)' any "$t/lease" STUB_COMPUTE_RC=42
  case_ 'any, failed partial process query names another card: takes the 3090' 0 'ok on the 3090 \(asked any\)' any "$t/lease" STUB_COMPUTE_RC=42 STUB_COMPUTE_UUID="$GPU_3090"
  case_ 'any, failed partial process query names the A6000: takes the 3090' 0 'ok on the 3090 \(asked any\)' any "$t/lease" STUB_COMPUTE_RC=42 STUB_COMPUTE_UUID="$GPU_A6000"
  case_ 'a6000 forced, process query fails: still runs on the A6000' 0 "ran on $GPU_A6000" a6000 "$t/lease" STUB_COMPUTE_RC=42
  case_ 'a6000 forced, lease cannot be tested: 70, named' 70 'cannot be tested' a6000 "$t/no-dir/lease"
  case_ 'a V4.1 load alone takes the V4.1 load lock and does not wait' 0 'ok on the 3090 \(asked 3090\), V4\.1 load lock' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  case_ 'BLOOMERY_GATE_V41_LOAD other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_V41_LOAD is 1' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=yes
  case_ 'BLOOMERY_GATE_STACKS not whole seconds: 64, named' 64 'BLOOMERY_GATE_STACKS is whole seconds' 3090 "$t/lease" BLOOMERY_GATE_STACKS=soon
  # The watch's quiet time only catches a hang here: the stub ends on its own.
  case_ 'BLOOMERY_GATE_STACKS set: the binary runs under the watch, its output through' 0 'ran on $' 3090 "$t/lease" BLOOMERY_GATE_STACKS=$fs
  case_ "a run closes with its own wall and exit" 0 '^gpu-gate\.sh: ok ran [0-9]+ s \(exit 0\)$' 3090 "$t/lease"
  case_ 'BLOOMERY_GATE_GDB other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_GDB is 1' 3090 "$t/lease" BLOOMERY_GATE_GDB=yes
  case_ 'BLOOMERY_GATE_GDB with BLOOMERY_GATE_STACKS: 64, named' 64 'set one' 3090 "$t/lease" BLOOMERY_GATE_GDB=1 BLOOMERY_GATE_STACKS=5
  # PIN(2026-10-31): PIN_3090 is gone: the Qwen3.8 serve gate's paged clause plans its rooms and its CLI on the gate card's bytes.
  cp "$t/tree/target/release/ok" "$t/tree/target/release/gate_qwen38_serve"
  gate_args=(--arm plain)
  GATE_BIN=gate_qwen38_serve case_ "the Qwen3.8 serve gate's plain arm runs on the A6000 in the real tier" 0 'ran on ' a6000 "$t/lease" BLOOMERY_TIER=real
  gate_args=()
  # The pack phase (tools/pack-for-driver.sh). First the lib's own decisions, in this process with its one outside read (the driver's
  # version) and its one outside command (the build bound's runner) replaced by recorders: what each kind of executable draws, and the
  # codes of a pack that fails. Then gpu-gate.sh's call of it, with a stand-in lib that records the call: it comes before any lock file
  # is opened and before BLOOMERY_BUILD_ONLY's early end is passed, and a pack that fails ends the call by its own code and runs nothing.
  # The ELF cases need objcopy and an ELF `true` (the box and Linux CI have both). pk-packed is pk-sm120 as oxart-pack.sh leaves it, its
  # `.oxart` renamed `.oxart.ptx` and a new `.oxart` beside it; that one still names the target here, so only the mark decides.
  local pk_elf='' pk_c pk_ct=$t/pk-ct
  for pk_c in /usr/bin/true /bin/true; do
    [ -x "$pk_c" ] && [ "$(head -c 4 "$pk_c" | od -An -tx1 | tr -d ' \n')" = 7f454c46 ] && { pk_elf=$pk_c; break; }
  done
  if [ -n "$pk_elf" ] && command -v objcopy > /dev/null; then
    # shellcheck source=tools/gate-bound.sh
    source "${self%/*}/gate-bound.sh"
    # shellcheck source=tools/pack-for-driver.sh
    source "${self%/*}/pack-for-driver.sh"
    printf '.target sm_86\n' > "$t/pk86.ptx"
    printf '// bundle\n.version 9.0\n.target sm_120a\n' > "$t/pk120.ptx"
    objcopy --add-section .oxart="$t/pk86.ptx" --set-section-flags .oxart=alloc,load,readonly,data,contents "$pk_elf" "$t/pk-sm86" 2> /dev/null
    objcopy --add-section .oxart="$t/pk120.ptx" --set-section-flags .oxart=alloc,load,readonly,data,contents "$pk_elf" "$t/pk-sm120" 2> /dev/null
    objcopy --rename-section .oxart=.oxart.ptx --add-section .oxart="$t/pk120.ptx" --set-section-flags .oxart=alloc,load,readonly,data,contents "$t/pk-sm120" "$t/pk-packed" 2> /dev/null
    cp "$pk_elf" "$t/pk-plain"
    mkdir -p "$pk_ct/release"
    : > "$pk_ct/release/oxart_ptxas"
    chmod +x "$pk_ct/release/oxart_ptxas"
    export CARGO_TARGET_DIR=$pk_ct
    local PK_DRIVER=13040 PK_RC=0 PK_BUILD_RC=0
    driver_cuda_version() {
      : > "$t/pk.driver"
      [ "$PK_DRIVER" != fail ] || { echo "pkcase: no libcuda to read the driver's CUDA version from" >&2; return 70; }
      echo "$PK_DRIVER"
    }
    pack_bounded() {
      printf '%s\n' "$*" >> "$t/pk.calls"
      case $1 in env) return "$PK_BUILD_RC" ;; *) return "$PK_RC" ;; esac
    }
    # pk <name> <executable> <want rc> <ERE>…: one pack_for_driver call; its output, then the count of commands it ran (the packer's
    # build and the pack) and whether it read the driver, are judged together (judge_all: an ERE that starts with ! must match nothing).
    pk() {
      local pkrc=0
      rm -f "$t/pk.calls" "$t/pk.driver"
      pack_for_driver pkcase "$2" > "$t/out" 2>&1 || pkrc=$?
      echo "ran=$([ -e "$t/pk.calls" ] && wc -l < "$t/pk.calls" | tr -d ' ' || echo 0) driver=$([ -e "$t/pk.driver" ] && echo read || echo unread)" >> "$t/out"
      [ ! -e "$t/pk.calls" ] || sed 's/^/ran: /' "$t/pk.calls" >> "$t/out"
      judge_all "pack: $1" "$pkrc" "$3" "${@:4}"
    }
    pk 'a shell script is no candidate: nothing read, nothing run, nothing said' "$t/tree/target/release/ok" 0 '^ran=0 driver=unread$' '!packing|not packed|GATE'
    pk 'an ELF with no .oxart is no candidate' "$t/pk-plain" 0 '^ran=0 driver=unread$' '!packing|not packed|GATE'
    pk 'sm_86 PTX is no candidate: the driver is not read' "$t/pk-sm86" 0 '^ran=0 driver=unread$' '!packing|not packed|GATE'
    PK_DRIVER=fail pk 'a packed binary passes before the driver is read' "$t/pk-packed" 0 '^ran=0 driver=unread$' '!packing|not packed|GATE'
    PK_DRIVER=13040 pk 'sm_120a PTX under CUDA 13.4 is left as it is' "$t/pk-sm120" 0 \
      'carries sm_120a PTX, driver CUDA 13\.4 is 13\.1 or later: not packed' '^ran=0 driver=read$' '!packing it|GATE'
    PK_DRIVER=13010 pk 'the first fixed driver itself is left as it is' "$t/pk-sm120" 0 'driver CUDA 13\.1 is 13\.1 or later: not packed' '^ran=0 driver=read$' '!packing it|GATE'
    PK_DRIVER=13000 pk 'sm_120a PTX under CUDA 13.0 packs it: oxart-pack.sh on the executable, no build' "$t/pk-sm120" 0 \
      'driver CUDA 13\.0 is before 13\.1: packing it' '^ran=1 driver=read$' "^ran: bash .*/tools/oxart-pack\\.sh $t/pk-sm120 $pk_ct/release/oxart_ptxas\$" '!GATE'
    PK_DRIVER=13000 PK_RC=65 pk "oxart-pack.sh's 65 (packed meanwhile, or nothing to pack) is no error" "$t/pk-sm120" 0 '^ran=1 driver=read$' '!GATE'
    PK_DRIVER=13000 PK_RC=1 pk 'a pack that fails is a named red in the pack phase, with its own code' "$t/pk-sm120" 1 \
      'GATE RED \(exit 1\) in the pack phase — no test ran'
    PK_DRIVER=13000 PK_RC=69 pk 'a missing tool is 69, named in the pack phase' "$t/pk-sm120" 69 'GATE RED \(exit 69\) in the pack phase — no test ran'
    PK_DRIVER=13000 PK_RC=124 pk 'a pack past the build bound is named a timeout of the pack phase' "$t/pk-sm120" 124 \
      'GATE TIMED OUT in the pack phase after the 2700s build bound \(exit 124' '!GATE RED'
    PK_DRIVER=fail pk 'an unreadable driver ends 70, named, before any pack' "$t/pk-sm120" 70 \
      'pkcase: no libcuda' 'GATE RED \(exit 70\) in the pack phase — no test ran' '^ran=0 driver=read$'
    rm "$pk_ct/release/oxart_ptxas"
    PK_DRIVER=13000 pk 'no oxart_ptxas: it is built first (plain cargo, in the repo), then the pack runs' "$t/pk-sm120" 0 \
      '^ran=2 driver=read$' '^ran: env -C .* cargo build --release -p bloomery-gpu-gates --bin oxart_ptxas$' '^ran: bash .*/tools/oxart-pack\.sh ' '!GATE'
    PK_DRIVER=13000 PK_BUILD_RC=3 pk "a failed build of oxart_ptxas ends the call with cargo's code and packs nothing" "$t/pk-sm120" 3 \
      'GATE RED \(exit 3\) in the pack phase — no test ran' '^ran=1 driver=read$' '!oxart-pack\.sh'
    unset CARGO_TARGET_DIR
    unset -f driver_cuda_version pack_bounded pk
    mkdir -p "$t/pkg"
    for bf in "$(dirname "$self")"/*; do [ "${bf##*/}" = gpu-gate.sh ] || [ "${bf##*/}" = pack-for-driver.sh ] || ln -sf "$bf" "$t/pkg/${bf##*/}"; done
    ln -sf "$self" "$t/pkg/gpu-gate.sh"
    printf '%s\n' 'pack_for_driver() {' '  echo "stand-in pack: $1 $2"' \
      '  [ -z "${STUB_PACK_RC:-}" ] || { echo "GATE RED (exit $STUB_PACK_RC) in the pack phase — no test ran" >&2; return "$STUB_PACK_RC"; }' '}' > "$t/pkg/pack-for-driver.sh"
    # pkg <name> <want rc> <ERE>…: one run of gpu-gate.sh over the stand-in lib, no lock file of its own before it (the environment of the
    # call, STUB_PACK_RC and the like, reaches the run); whether the run opened its card lock file is judged beside the output.
    pkg() {
      local pkn=$1 pkw=$2 pkrc=0
      shift 2
      rm -f "$t/gate.lock" "$t/gate-a6000.lock" "$t/v41-load.lock"
      self=$t/pkg/gpu-gate.sh gate "$t/out" 3090 "$t/lease" || pkrc=$?
      echo "gate.lock: $([ -e "$t/gate.lock" ] && echo opened || echo never-opened)" >> "$t/out"
      judge_all "pack call: $pkn" "$pkrc" "$pkw" "$@"
    }
    pkg 'the pack is called with the runner and the executable, and the binary then runs, its locks open after it' 0 \
      '^stand-in pack: gpu-gate\.sh \./target/release/ok$' 'ran on' '^gate\.lock: opened$' '!GATE RED'
    STUB_PACK_RC=7 pkg 'a pack that fails ends the call by its own code before any lock file is opened, no binary run' 7 \
      'GATE RED \(exit 7\) in the pack phase — no test ran' '^gate\.lock: never-opened$' '!ran on'
    STUB_PACK_RC=7 BLOOMERY_BUILD_ONLY=1 pkg 'BLOOMERY_BUILD_ONLY ends the call before the pack phase' 0 \
      'ok is built; nothing runs, no lock is taken' '^gate\.lock: never-opened$' '!stand-in pack|GATE RED'
    unset -f pkg
  else
    echo "skip pack-for-driver cases: no ELF true or no objcopy on this machine (the box and Linux CI run them)"
  fi
  # Events, not seconds. A run in the background is held by what it waits for and released by what the test saw it do, so no case reads
  # the clock: the stub binary says `up` on STUB_READY once its run holds its locks and then blocks on STUB_HOLD until the test writes a
  # line; a waiting run's own output (read through a FIFO by relay) says it waits. The FIFOs are held open read-write by this shell
  # (descriptors 3: events, 4: the stub's release), so neither side blocks on opening one and a test that ends releases the stub by EOF.
  # A helper waiter's two bounds are a safety cap of $cap s (GATE_TEST_CAP), under gate()'s own bound for it so a waiter that hits it
  # ends 75, not 124: the test ends its life by an event, and one that hits the cap is a red by name (helper_cap). The cap is a life
  # long enough for every event the test waits for beside it, each late on a loaded host. The cases that judge a bound keep the
  # runner's own 4 s and 6 s.
  fi
  fifos() {
    exec 3>&- 4>&-
    rm -f "$t/ready" "$t/hold"
    mkfifo "$t/ready" "$t/hold"
    exec 3<> "$t/ready" 4<> "$t/hold"
  }
  # streams <fifo…>: fresh FIFOs a run's output goes through.
  streams() {
    rm -f "$@"
    mkfifo "$@"
  }
  # event <case>: blocks until one event is on descriptor 3; a run that ends without it writes one itself (relay), and fs s of silence is a hang.
  event() { read -t "$fs" -r _ <&3 || { n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $1: the event never came"; }; }
  # relay <fifo> <out file> <string> <fd> [<string> <fd>]: reads a run's output from <fifo> into <out file> as it comes; the first line that
  # holds <string> writes one line to descriptor <fd>, and so does the end of the output when none did (a waiter never outlives its run).
  relay() {
    local f=$1 o=$2 str=$3 fd=$4 str2=${5:-} fd2=${6:-} line seen=0 seen2=0
    : > "$o"
    while IFS= read -r line; do
      printf '%s\n' "$line" >> "$o"
      case $line in *"$str"*) if [ "$seen" = 0 ]; then echo go >&"$fd"; seen=1; fi ;; esac
      if [ -n "$str2" ]; then
        case $line in *"$str2"*) if [ "$seen2" = 0 ]; then echo go >&"$fd2"; seen2=1; fi ;; esac
      fi
    done < "$f"
    [ "$seen" = 1 ] || echo go >&"$fd"
    [ -z "$str2" ] || [ "$seen2" = 1 ] || echo go >&"$fd2"
  }
  if gg_part 1; then
  # Two V4.1 loads at once, on the two cards: the second waits for the first, names it, and counts the wait. The first holds the load
  # lock until the second has scanned for its holder (lease_holders prints once its scan of /proc is done, so its first `[lease]   `
  # line); the second starts once the first holds it.
  fifos
  gate "$t/first" a6000 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 STUB_READY="$t/ready" STUB_HOLD="$t/hold" &
  p1=$!
  event 'two V4.1 loads at once'
  streams "$t/s2"
  gate "$t/s2" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_LEASE_PROC=/proc GATE_TEST_CAP="$cap" &
  p2=$!
  relay "$t/s2" "$t/out" '[lease]   ' 4
  rc=0 rc1=0
  wait "$p2" || rc=$?
  wait "$p1" || rc1=$?
  cat "$t/first" >> "$t/out"
  judge 'two V4.1 loads at once: the second waits, names the holder, counts the wait' "$rc" 0 \
    'waits for the V4\.1 load lock .*holding no gate lock'
  if [ -d /proc/self ]; then
    judge '  … its wait line names the first run as the holder' "$rc" 0 '\[lease\] +pid [0-9]+ holds it'
  else
    echo "skip   … its wait line names the first run as the holder: no /proc here (lease_holders reads it)"
  fi
  judge '  … and the waited line the batch runner subtracts' "$rc" 0 '^gpu-gate\.sh: waited [0-9]+ s for the V4\.1 load lock$'
  cp "$t/first" "$t/out"
  judge '  … the first ran on the A6000 without waiting' "$rc1" 0 "ran on $GPU_A6000" 'waits for the V4\.1'
  # A run that is not a V4.1 load does not wait for one: the load lock is held by the test throughout, so a run that waited for it
  # would end 75 at the bound with the lock named, never 0.
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  rc=0
  gate "$t/out" 3090 "$t/lease" || rc=$?
  flock -u 6
  exec 6>&-
  judge 'a run that is not a V4.1 load does not wait for one' "$rc" 0 'ok on the 3090 \(asked 3090\)' 'V4\.1'
  # gate()'s bound ends a held run, under the watch too. The stub writes `up` once its TERM handler is installed
  # and then holds; that line, then both timeout monitors asleep, is the event, and SIGALRM to the outer timeout is the bound
  # (the signal its own timer sends). Each monitor sleeps once its fork has returned: GNU coreutils timeout before 9.5
  # (Ubuntu 24.04's 9.4) ends 128+n without signalling a child that ran before its fork returned. A clock from the
  # spawn can run out before the stub holds, and says nothing of the contract. The failsafe
  # bound ($fs) only catches a hang: a run that never parks and never exits has no exit to wait on. The handler exits
  # 7 with no sleep. The runner's cleanup sends KILL at LEASE_STOP_GRACE-2 s (grace 9, 2 s kept so the KILL and reap
  # finish before an enclosing kill-after of 10 s), and a sleep in the handler is a clock that KILL can win, so the
  # runner records 137. The judge waits on the run's exit and on the output's `end` line. A stub still held is let go.
  local w ended left sp ip lk tp line ready outer_state
  for w in '' "BLOOMERY_GATE_STACKS=$((fs + 60))"; do
    fifos
    streams "$t/s1"
    rm -f "$t/stub.pid" "$t/timeout.pid"
    rc=0 ready=no ended=no line=
    # shellcheck disable=SC2086 # w is a K=V word or none
    GATE_TIMEOUT_PID=$t/timeout.pid GB=$fs GK=$fs \
      gate "$t/s1" 3090 "$t/lease" STUB_PID="$t/stub.pid" STUB_TERM=7 STUB_READY="$t/ready" STUB_HOLD="$t/hold" $w &
    p1=$!
    { cat "$t/s1" > "$t/first"; echo end >&3; } &
    r1=$!
    if read -t "$fs" -r line <&3; then
      if [ "$line" = up ]; then ready=yes
      elif [ "$line" = end ]; then ended=yes
      fi
    fi
    if [ "$ready" = yes ] && [ -s "$t/timeout.pid" ]; then
      read -r tp < "$t/timeout.pid" || tp=
      since=$SECONDS
      until { cancel_ready "$t/stub.pid" &&
        outer_state=$(ps -o stat= -p "$tp" 2> /dev/null) &&
        [[ $outer_state = *S* ]]; } ||
        [ $((SECONDS - since)) -ge "$fs" ]; do
        sleep 0.05
      done
      [ -z "$tp" ] || kill -s ALRM -- "$tp" 2> /dev/null || true
    fi
    wait "$p1" || rc=$?
    if [ "$ready" = yes ] && read -t "$fs" -r line <&3; then
      [ "$line" = end ] && ended=yes
    fi
    left=
    if [ -s "$t/stub.pid" ]; then
      read -r sp ip < "$t/stub.pid"
      if kill -0 "$sp" 2> /dev/null; then left="$left stub"; fi
      if kill -0 "$ip" 2> /dev/null; then left="$left timeout"; fi
    fi
    lk=free
    flock -n "$t/gate.lock" true || lk=held
    echo go >&4
    wait "$r1" || true
    echo "stub ran: $([ -s "$t/stub.pid" ] && echo yes || echo no), left:${left:- none}, output ended: $ended, gate lock: $lk" > "$t/out"
    sed 's/^/run: /' "$t/first" >> "$t/out"
    judge_all "gate()'s bound ends a held run${w:+ under the watch} with its binary, output and lock, the binary's code kept" "$rc" 124 \
      'stub ran: yes, left: none, output ended: yes, gate lock: free$' '^run: gpu-gate\.sh: ok ran [0-9]+ s \(exit 7\)$'
  done
  # A PID-only TERM while the watch sleeps must wait for a TERM-ignoring command's KILL.
  # The sleep fixture holds that foreground wait until the TERM, then delays its trap by two seconds.
  # Descriptor 5 keeps the output FIFO inherited even while the watch redirects its command's output.
  # shellcheck source=tools/gate-bound.sh
  source "${self%/*}/gate-bound.sh"
  local ready signalled before state child lock forward child_status want
  mkdir "$t/stack-bin" || return 70
  cancel_timeout "$t/stack-bin/timeout" || return 70
  cat > "$t/stack-bin/sleep" << 'SH' || return 70
#!/bin/sh
if [ "$#" = 1 ] && [ "$1" = 1 ] && [ -s "$SW_CMD" ]; then
  echo "$$ $PPID" > "$SW_SLEEP" || exit 70
  read -r _ < "$SW_HOLD" || exit 70
  exec /bin/sleep 2
fi
exec /bin/sleep "$@"
SH
  printf '%s\n' '#!/bin/sh' 'trap "" TERM INT HUP' 'echo "$$ $PPID" > "$SW_CMD" || exit 70' \
    'exec /bin/sleep 120' > "$t/tree/target/release/held" || return 70
  chmod +x "$t/stack-bin/timeout" "$t/stack-bin/sleep" "$t/tree/target/release/held" || return 70
  fifos || return 70
  streams "$t/stack-output" || return 70
  { cat "$t/stack-output" > "$t/first" && echo end >&3; } &
  r1=$!
  (
    cd "$t/tree" || exit 70
    exec 5>&1 || exit 70
    exec "${gate_env[@]}" PATH="$t/stack-bin:$t/bin:$PATH" \
      BLOOMERY_GATE_CARD=3090 BLOOMERY_LEASE_LOCK="$t/lease" BLOOMERY_LEASE_PROC="$t/proc" \
      BLOOMERY_GATE_STACKS=60 BLOOMERY_GATE_BOUND=$((fs + 60)) BLOOMERY_GATE_V41_LOAD=1 \
      SW_CMD="$t/stack-command.pid" SW_SLEEP="$t/stack-sleep.pid" SW_HOLD="$t/hold" \
      bash "$self" --test-locks "$t" held
  ) > "$t/stack-output" 2>&1 &
  p1=$! ready=no
  # The runner is waited for while it lives, fs s at most: a run that ends before it is ready is the red below.
  for ((child = SECONDS; SECONDS - child < fs; )); do
    if cancel_ready "$t/stack-command.pid" && cancel_ready "$t/stack-sleep.pid"; then ready=yes; break; fi
    kill -0 "$p1" 2> /dev/null || break
    sleep 0.1
  done
  before=''
  for lock in gate v41-load; do
    if flock -n "$t/$lock.lock" true; then before="$before $lock=free"; else before="$before $lock=held"; fi
  done
  signalled=yes
  kill -TERM "$p1" 2> /dev/null || signalled=no
  echo go >&4 || return 70
  rc=0; wait "$p1" || rc=$?
  left='' sp='' ip=''
  if [ -s "$t/stack-command.pid" ]; then
    read -r sp ip < "$t/stack-command.pid" || return 70
    for child in "$sp" "$ip"; do
      if state=$(ps -o stat= -p "$child"); then
        [[ $state =~ ^[[:space:]]*Z ]] || left="$left $child"
      else
        [ "$?" = 1 ] || return 70
      fi
    done
  fi
  lk=''
  for lock in gate v41-load; do
    if flock -n "$t/$lock.lock" true; then lk="$lk $lock=free"; else lk="$lk $lock=held"; fi
  done
  ended=no
  if read -t "$fs" -r _ <&3; then ended=yes; fi
  echo "ready: $ready, TERM sent: $signalled, locks before:$before" > "$t/out" || return 70
  echo "left:${left:- none}, output ended: $ended, locks after:$lk" >> "$t/out" || return 70
  sed 's/^/run: /' "$t/first" >> "$t/out" || return 70
  judge_all 'a TERM to the sleeping watch runner waits for KILL, output EOF and both locks' "$rc" 143 \
    '^ready: yes, TERM sent: yes, locks before: gate=held v41-load=held$' \
    '^left: none, output ended: yes, locks after: gate=free v41-load=free$'
  # Observe before cleanup, so an orphaned group cannot turn a failure green.
  [ -z "$ip" ] || lease_stop "$ip" || return 70
  wait "$r1" || return 70
  # The stub forwards once; GNU timeout may send TERM to both its child and the child's group.
  # One cancellation lets a command drain on its first TERM; a second TERM aborts that drain.
  for w in forwarding nonforwarding; do
    rc=0 want=0 forward=1
    if [ "$w" = nonforwarding ]; then forward=0; want=143; fi
    CANCEL_TIMEOUT_FORWARD=$forward PATH="$t/stack-bin:$PATH" bash -c '
      source "$1"
      export STOP_RUNNER=$$
      run_bounded timeout --kill-after=10 "$2" python3 -
    ' _ "${self%/*}/gate-bound.sh" "$fs" > "$t/out" 2>&1 << 'PY' || rc=$?
import os
import signal
import sys
import time

def abort(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(1, b"second TERM: abort\n")
    os._exit(1)

def drain(*_):
    signal.signal(signal.SIGTERM, abort)
    time.sleep(0.3)
    print("drained: exit 0", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, drain)
os.kill(int(os.environ["STOP_RUNNER"]), signal.SIGTERM)
while True:
    signal.pause()
PY
    judge_all "one cancellation drains a command under a $w monitor" "$rc" "$want" \
      '^drained: exit 0$' '!second TERM: abort'
  done
  # Inject TERM at the monitor's child-handle and wait boundaries.
  cat > "$t/term-window.py" << 'PY' || return 70
import os
import runpy
import select
import signal
import subprocess
import sys

fixture, phase, failsafe = sys.argv[1], sys.argv[2], float(sys.argv[3])
read_fd, write_fd = os.pipe()
os.set_inheritable(write_fd, True)
spawned = []
fired = False
original = subprocess.Popen


def term():
    global fired
    if not fired:
        fired = True
        os.kill(os.getpid(), signal.SIGTERM)


def read_state(expected):
    assert select.select([read_fd], [], [], failsafe)[0], "child state timed out"
    assert os.read(read_fd, 1) == expected, "unexpected child state"


class WindowChild(original):
    def __init__(self, *args, **kwargs):
        if phase == "before-spawn":
            term()
        super().__init__(*args, **kwargs)
        spawned.append(self)
        os.close(write_fd)
        read_state(b"r")
        if phase == "before-handle":
            term()

    def wait(self, *args, **kwargs):
        if phase in ("before-wait", "second-term"):
            term()
            if phase == "second-term":
                read_state(b"d")
                os.kill(os.getpid(), signal.SIGTERM)
        rc = super().wait(*args, **kwargs)
        if phase == "after-wait":
            term()
        return rc

    def _wait(self, *args, **kwargs):
        if phase == "during-wait":
            term()
        return super()._wait(*args, **kwargs)


subprocess.Popen = WindowChild
sys.argv = [fixture, str(int(failsafe)), sys.executable, "-c", '''
import os, signal, sys, time
fd = int(sys.argv[1])
def abort(*_):
    os.write(1, b"second TERM: abort\\n")
    os._exit(1)
def drain(*_):
    signal.signal(signal.SIGTERM, abort)
    os.write(fd, b"d")
    if sys.argv[2] == "second-term":
        while True:
            signal.pause()
    time.sleep(0.1)
    print("drained: exit 0", flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, drain)
os.write(fd, b"r")
if sys.argv[2] == "after-wait":
    print("completed: exit 7", flush=True)
    sys.exit(7)
while True:
    signal.pause()
''', str(write_fd), phase]
try:
    runpy.run_path(fixture, run_name="__main__")
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for child in spawned:
        if child.poll() is None:
            child.kill()
        original.wait(child, timeout=failsafe)
PY
  for w in before-spawn before-handle before-wait during-wait after-wait second-term; do
    rc=0 want=0 state='^drained: exit 0$'
    case $w in
      after-wait) want=7; state='^completed: exit 7$' ;;
      second-term) want=1; state='^second TERM: abort$' ;;
    esac
    CANCEL_TIMEOUT_FORWARD=1 python3 "$t/term-window.py" "$t/stack-bin/timeout" "$w" "$fs" > "$t/out" 2>&1 || rc=$?
    judge "monitor TERM at $w keeps the child's status" "$rc" "$want" "$state"
  done
  # A failed process read during cancellation stays an error even if the child ends successfully;
  # a nonzero child status still wins over the cleanup error.
  for child_status in 0 7; do
    rc=0 want=70
    [ "$child_status" = 0 ] || want=$child_status
    (
      # shellcheck source=tools/gate-bound.sh
      source "${self%/*}/gate-bound.sh"
      ps() { sleep 0.1; return 1; }
      run_bounded python3 - "$child_status" "$fs" << 'PY'
import os
import signal
import sys

def stop(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.exit(int(sys.argv[1]))

os.setpgid(0, 0)
signal.signal(signal.SIGTERM, stop)
signal.alarm(int(sys.argv[2]))
os.kill(os.getppid(), signal.SIGTERM)
while True:
    signal.pause()
PY
    ) > "$t/out" 2>&1 || rc=$?
    judge "a failed process read after TERM keeps cleanup failure (child $child_status)" "$rc" "$want" \
      'cannot read processes while stopping'
  done
  # The two waits, counted apart: the test holds the load lock and the 3090's gate lock. A loader (the head of a queue of one) waits for
  # the load lock, takes it once the test lets it go and then, holding it, waits for its card: each wait is one `waited` line of its own
  # (the load lock's seconds end where the card's begin), and the card wait says it holds the load lock.
  fifos
  fi
  qreset() { rm -rf "$t/v41-queue.d" "$t/v41-queue.seq" "$t/order"; }
  if gg_part 1; then
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  exec 5> "$t/gate.lock"
  flock -x 5 || { echo "FAIL: the test cannot hold the 3090 gate lock"; return 1; }
  streams "$t/s1"
  gate "$t/s1" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 GATE_TEST_CAP="$cap" &
  p1=$!
  relay "$t/s1" "$t/first" 'queued at' 3 'waits for the gate lock' 3 &
  r1=$!
  event 'waited lines: queued'
  flock -u 6
  event 'waited lines: the card wait'
  flock -u 5
  exec 5>&-
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  wait "$r1" || true
  cp "$t/first" "$t/out"
  judge_all 'the load lock and the card lock are two waits, each in its own waited line' "$rc" 0 \
    '^gpu-gate\.sh: waited [0-9]+ s for the V4\.1 load lock$' '^gpu-gate\.sh: waited [0-9]+ s for the gate lock \(3090\)$' \
    'waits for the gate lock \(3090\), holding the V4\.1 load lock' 'ok on the 3090 \(asked 3090\), V4\.1 load lock$'
  # Taken together: the load lock is held (by the test) and two loaders wait on it, one asking `any` each. A loader that held
  # the card it had got while it waited would leave an `any` gate that is no load with no card, and with the load lock held until that
  # gate ends it could only end 75 at the card bound; with neither held, that gate takes a card. Each loader's seconds are its
  # load-lock wait alone. The gate starts once both loaders have said they wait.
  fifos
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  streams "$t/s1" "$t/s2"
  gate "$t/s1" any "$t/lease" BLOOMERY_GATE_V41_LOAD=1 GATE_TEST_CAP="$cap" &
  p1=$!
  gate "$t/s2" any "$t/lease" BLOOMERY_GATE_V41_LOAD=1 GATE_TEST_CAP="$cap" &
  p2=$!
  relay "$t/s1" "$t/w1" 'queued at' 3 &
  r1=$!
  relay "$t/s2" "$t/w2" 'queued at' 3 &
  r2=$!
  event 'together'
  event 'together'
  rc=0
  gate "$t/out" any "$t/lease" || rc=$?
  judge 'together: two loaders queued on the load lock hold no card, an any gate takes one at once' "$rc" 0 'ran on' 'no gate lock'
  judge '  … and takes it, not a queue' "$rc" 0 'ok on the (3090|a6000) \(asked any\)'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  wait "$r1" || true
  wait "$r2" || true
  # Each loader counts its own load-lock wait. A card wait may show too: at the instant the lock frees both ask for the one card
  # `any` picks, and the loser waits for it honestly, so no line here says a card was not waited for.
  cp "$t/w1" "$t/out"
  judge '  … the first queued loader runs once the lock frees, counting its load-lock wait' "$rc" 0 'waited [0-9]+ s for the V4\.1 load lock'
  cp "$t/w2" "$t/out"
  judge '  … and the second' "$rc" 0 'waited [0-9]+ s for the V4\.1 load lock'
  # The same for a loader of both cards: it took both card locks, then waited for the load lock holding them.
  fifos
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  streams "$t/s1"
  gate "$t/s1" both "$t/lease" BLOOMERY_BOX_CARD=both BLOOMERY_GATE_V41_LOAD=1 GATE_TEST_CAP="$cap" &
  p1=$!
  relay "$t/s1" "$t/w1" 'waits for the V4.1 load lock' 3 &
  r1=$!
  event 'together (both cards)'
  rc=0
  gate "$t/out" any "$t/lease" || rc=$?
  judge 'together: a two-card loader queued on the load lock holds no card, an any gate takes one at once' "$rc" 0 'ran on' 'no gate lock'
  flock -u 6
  exec 6>&-
  rc=0
  wait "$p1" || rc=$?
  wait "$r1" || true
  cp "$t/w1" "$t/out"
  judge '  … and the loader runs on both cards once the lock frees' "$rc" 0 'on both cards, both gate locks \(asked both\), V4\.1 load lock'
  # The V4.1 load queue. A paced loader (pgate) polls only when the test grants it a poll (the pace function), so the order loaders are
  # served in is the queue's and never the clock's: the test holds the load lock while loaders arrive, lets it go, and grants the polls in
  # an order a run that tried the lock whether or not it was the head would win by. Each grant ends with the loader's next sign on its ack
  # FIFO: `polled` (it waited) or `exit` (it took the locks, ran the stub, and ended); the stub's marker file says in what order.
  # pace_open <k>: loader k's FIFO pair, held open read-write here (descriptors 20+2k grants, 21+2k signs).
  fi
  pace_open() {
    local g=$((20 + 2 * $1)) a=$((21 + 2 * $1))
    rm -f "$t/pc$1.go" "$t/pc$1.ack"
    mkfifo "$t/pc$1.go" "$t/pc$1.ack"
    ENDED[$1]=0
    LPID[$1]=
    eval "exec $g<> '$t/pc$1.go' $a<> '$t/pc$1.ack'"
  }
  pace_close() {
    local g=$((20 + 2 * $1)) a=$((21 + 2 * $1))
    eval "exec $g>&- $a>&-"
  }
  # ack <k> <case>: ACK is loader k's next sign. The sign is an event: the wait goes on while the loader's run lives (a loader late to
  # start on a loaded host is waited for), and ends in a red by name when the run is gone without a sign or has been silent for fs s (a
  # hang). A loader that has signed `exit`, or has failed to sign, is gone: its next sign is `gone` at once, not a wait for one that
  # cannot come. The `exit` sign comes from the EXIT trap, before the process lets go of its locks (the V4.1 load lock, the card lock),
  # so `exit` is taken as an end only once the run's pid is gone: a grant to the next loader right after it would find the lock still
  # held, and that loader would poll instead of run (a sign short, then a grant it never gets).
  ack() {
    local t0=$SECONDS why=
    ACK=
    if [ "${ENDED[$1]:-0}" = 1 ]; then
      ACK=gone
      return 0
    fi
    until read -t 5 -r ACK <&"$((21 + 2 * $1))"; do
      if ! kill -0 "${LPID[$1]:-0}" 2> /dev/null; then
        # The run is gone: a sign it wrote before it went is still in the FIFO.
        read -t 5 -r ACK <&"$((21 + 2 * $1))" || why="its run ended without one"
        break
      fi
      [ $((SECONDS - t0)) -lt "$fs" ] || { why="it was silent for $fs s"; break; }
    done
    if [ -n "$why" ]; then
      n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $2: loader $1 gave no sign ($why)"
      ACK=none
      ENDED[$1]=1
      return 0
    fi
    [ "$ACK" != exit ] || ENDED[$1]=1
    if [ "$ACK" = exit ] && [ -n "${LPID[$1]:-}" ]; then
      t0=$SECONDS
      while kill -0 "${LPID[$1]}" 2> /dev/null; do
        [ $((SECONDS - t0)) -lt "$fs" ] || { n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $2: loader $1 signed exit and was still there after $fs s"; break; }
        sleep 0.05
      done
    fi
  }
  # grant <k> <case>: lets loader k poll once and takes its sign.
  grant() {
    echo go >&"$((20 + 2 * $1))"
    ack "$@"
  }
  # pgate <k> <id> <class> <card> [K=V…]: starts paced loader k (marker <id>, output $t/q<k>); PG is the pid of the run. The job's own stderr is
  # dropped: it is where bash reports a loader the test kills (`Killed`, `Terminated`); the loader's output goes to its file.
  pgate() {
    local k=$1 id=$2 cl=$3 c=$4
    shift 4
    pace_open "$k"
    gate "$t/q$k" "$c" "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_V41_CLASS="$cl" GATE_TEST_PACE="$t/pc$k" GATE_TEST_CAP="$cap" \
      STUB_ORDER="$t/order" STUB_ID="$id" "$@" 2> /dev/null &
    PG=$!
    LPID[$k]=$PG
  }
  # markers: the stub's markers in the order it wrote them, on one line.
  markers() { printf 'markers: %s\n' "$(tr '\n' ' ' < "$t/order" 2> /dev/null)"; }
  if gg_part 2; then
  # Three loaders arrive in the order A, B, C while the test holds the load lock; the view lists them in order; once the lock is free the
  # polls are granted C, B, A, B, C: only the head may take it, so C and B poll in vain and the markers are A, B, C.
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  pgate 0 A 2 any BLOOMERY_V41_PRED_S=11
  p1=$PG
  ack 0 'queue order'
  pgate 1 B 2 any BLOOMERY_V41_PRED_S=22
  p2=$PG
  ack 1 'queue order'
  pgate 2 C 2 any BLOOMERY_V41_PRED_S=33
  p3=$PG
  ack 2 'queue order'
  rc=0
  bash "$self" --test-locks "$t" --v41-queue > "$t/out" 2>&1 || rc=$?
  echo "order: $(grep -o 'pred_s=[0-9]*' "$t/out" | tr '\n' ' ')" >> "$t/out"
  judge_all '--v41-queue lists the live tickets in order, one line each with the age' "$rc" 0 \
    '^2-0000000001-[0-9]+ owner=tree pid=[0-9]+ start=[^ ]+ since=[0-9]+ cards=any item=ok tier=real pred_s=11 age=[0-9]+$' \
    '^2-0000000002-[0-9]+ .* pred_s=22 age=[0-9]+$' '^2-0000000003-[0-9]+ .* pred_s=33 age=[0-9]+$' 'order: pred_s=11 pred_s=22 pred_s=33 $'
  cp "$t/q2" "$t/out"
  judge_all '  … a queued loader says its place and the head' 0 0 'gpu-gate\.sh: ok queued at 3 of 3 for the V4\.1 load lock \(head: ok tree, [0-9]+ s\)'
  flock -u 6
  exec 6>&-
  grant 2 'queue order'
  s1=$ACK
  grant 1 'queue order'
  s2=$ACK
  grant 0 'queue order'
  s3=$ACK
  grant 1 'queue order'
  s4=$ACK
  grant 2 'queue order'
  s5=$ACK
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  wait "$p3" || rc=$?
  echo "signs: $s1 $s2 $s3 $s4 $s5" > "$t/out"
  markers >> "$t/out"
  judge_all 'queue order: loaders arriving A, B, C are served A, B, C, however the polls fall' "$rc" 0 'signs: polled polled exit exit exit$' 'markers: A B C $'
  pace_close 0; pace_close 1; pace_close 2
  # Class: a class-1 loader that arrives after a class-2 one is served first.
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  pgate 0 X 2 any
  p1=$PG
  ack 0 'queue class'
  pgate 1 Y 1 any
  p2=$PG
  ack 1 'queue class'
  rc=0
  bash "$self" --test-locks "$t" --v41-queue > "$t/out" 2>&1 || rc=$?
  echo "order: $(cut -d- -f1,2 "$t/out" | tr '\n' ' ')" >> "$t/out"
  judge_all '--v41-queue puts a class-1 ticket ahead of an earlier class-2 one' "$rc" 0 'order: 1-0000000002 2-0000000001 $'
  flock -u 6
  exec 6>&-
  grant 0 'queue class'
  s1=$ACK
  grant 1 'queue class'
  s2=$ACK
  grant 0 'queue class'
  s3=$ACK
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  echo "signs: $s1 $s2 $s3" > "$t/out"
  markers >> "$t/out"
  judge_all 'queue class: a class-1 loader arriving after a class-2 one is served first' "$rc" 0 'signs: polled exit exit$' 'markers: Y X $'
  pace_close 0; pace_close 1
  # The exit sign is the trap's, written before the process lets go of its locks: loader A lingers 2 s in its trap after signing, holding the
  # V4.1 load lock, and B is granted its poll the moment A's sign is taken. B must find the lock free (ack waits for A to be gone), run, and
  # sign exit: a short sign here is the CI's 'queue order' / 'queue class' rc 124. Drain grants keep a failing run from waiting out the bound.
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  pgate 0 A 2 any GATE_TEST_LINGER=2
  p1=$PG
  ack 0 'exit sign'
  pgate 1 B 2 any
  p2=$PG
  ack 1 'exit sign'
  flock -u 6
  exec 6>&-
  grant 0 'exit sign'
  s1=$ACK
  grant 1 'exit sign'
  s2=$ACK
  for r1 in 1 2 3 4 5 6; do
    case $s2 in *polled) ;; *) break ;; esac
    sleep 0.5
    grant 1 'exit sign'
    s2="$s2 $ACK"
  done
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  echo "signs: $s1 $s2" > "$t/out"
  markers >> "$t/out"
  judge_all 'exit sign: a loader granted right after another signed exit finds the lock free and runs' "$rc" 0 'signs: exit exit$' 'markers: A B $'
  pace_close 0; pace_close 1
  # Dead tickets block nobody: a loader killed outright leaves its ticket (no trap ran), and a ticket whose pid lives on with another start
  # time (a reused pid) is a dead one too. A loader arriving behind both removes them on its first poll, says so, and is the head.
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  pgate 0 K 2 any
  p1=$PG
  ack 0 'dead tickets'
  kpid=$(sed -n 's/.* pid=\([0-9]*\) .*/\1/p' "$t"/v41-queue.d/2-*)
  kill -9 "$kpid"
  wait "$p1" || true
  printf 'owner=reuse pid=%s start=1 since=%s cards=any item=reuse tier=real pred_s=-\n' "$$" "$(date +%s)" > "$t/v41-queue.d/2-0000000000-$$"
  pgate 1 L 2 any
  p2=$PG
  ack 1 'dead tickets'
  cp "$t/q1" "$t/out"
  echo "tickets left: $(ls "$t/v41-queue.d" | wc -l | tr -d ' ')" >> "$t/out"
  judge_all 'dead tickets: a killed loader and a reused pid are removed, named, and the next loader is the head' 0 0 \
    'gpu-gate\.sh: removed the dead V4\.1 queue ticket 2-0000000001-[0-9]+ \(pid [0-9]+ is gone\)' \
    'gpu-gate\.sh: removed the dead V4\.1 queue ticket 2-0000000000-[0-9]+ \(pid [0-9]+ is another process now' \
    'queued at 1 of 1 for the V4\.1 load lock' 'tickets left: 1$'
  flock -u 6
  exec 6>&-
  grant 1 'dead tickets'
  s1=$ACK
  rc=0
  wait "$p2" || rc=$?
  echo "signs: $s1" > "$t/out"
  markers >> "$t/out"
  judge_all '  … and is served' "$rc" 0 'signs: exit$' 'markers: L $'
  pace_close 0; pace_close 1
  # The head that waits for its card holds the load lock: a loader of the 3090 queues on the card the test holds, the load lock is held
  # (the probe is shared, so it never meets a poll of the loader) and a second loader, asking `any` with the A6000 idle, stays queued
  # and holds no card; once the card is free the head runs first, then the second.
  qreset
  exec 6> "$t/gate.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the 3090 gate lock"; return 1; }
  pgate 0 H 2 3090
  p1=$PG
  ack 0 'head holds the load lock'
  prc=0
  flock -s -n "$t/v41-load.lock" true || prc=$?
  pgate 1 B 2 any
  p2=$PG
  ack 1 'head holds the load lock'
  pra=0
  flock -n "$t/gate-a6000.lock" true || pra=$?
  echo "probe load lock rc=$prc a6000 lock rc=$pra" > "$t/out"
  cat "$t/q0" "$t/q1" >> "$t/out"
  judge_all 'lock order: the head waiting for its card holds the load lock, the loader behind it takes nothing' 0 0 \
    'probe load lock rc=1 a6000 lock rc=0$' 'waits for the gate lock \(3090\), holding the V4\.1 load lock' \
    'queued at 2 of 2 for the V4\.1 load lock \(head: ok tree, [0-9]+ s\)'
  flock -u 6
  exec 6>&-
  grant 0 'head holds the load lock'
  s1=$ACK
  grant 1 'head holds the load lock'
  s2=$ACK
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  echo "signs: $s1 $s2" > "$t/out"
  markers >> "$t/out"
  judge_all '  … and is served first, the second loader after it' "$rc" 0 'signs: exit exit$' 'markers: H B $'
  pace_close 0; pace_close 1
  qreset
  # The batch hold stays above every class. A head holding the load lock for its card lets it go when another owner's hold goes up, and a
  # loader the hold holds back is no head for the batch's own loader, which passes the hold (its owner is the hold's) and arrives behind it
  # in the same class: the batch's loader is served first, the held-back ticket keeps its name (its place) meanwhile, and it is served once
  # the hold is down.
  qreset
  exec 6> "$t/gate.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the 3090 gate lock"; return 1; }
  pgate 0 H 2 3090
  p1=$PG
  ack 0 'hold over the queue'
  tk0=$(ls "$t/v41-queue.d")
  printf 'owner=other since=%s\n' "$(date +%s)" > "$t/batch.gpuhold"
  grant 0 'hold over the queue'
  s1=$ACK
  prc=0
  flock -s -n "$t/v41-load.lock" true || prc=$?
  pgate 1 Q 2 any BLOOMERY_BATCH_OWNER=other
  p2=$PG
  ack 1 'hold over the queue'
  s2=$ACK
  tk1=$(ls "$t/v41-queue.d")
  rm -f "$t/batch.gpuhold"
  grant 0 'hold over the queue'
  s3=$ACK
  flock -u 6
  exec 6>&-
  grant 0 'hold over the queue'
  s4=$ACK
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  echo "signs: $s1 $s2 $s3 $s4 probe load lock rc=$prc" > "$t/out"
  echo "held-back ticket before and after the batch's loader: $([ "$tk0" = "$tk1" ] && [ -n "$tk0" ] && echo same || echo "moved ($tk0 / $tk1)")" >> "$t/out"
  markers >> "$t/out"
  judge_all 'the batch hold: a head lets the load lock go and a held-back ticket is no head for the batch'"'"'s own loader' "$rc" 0 \
    'signs: polled exit polled exit probe load lock rc=0$' 'held-back ticket before and after the batch.s loader: same$' 'markers: Q H $'
  pace_close 0; pace_close 1
  # The sitting hold holds back tickets as the batch hold does: a loader it stops (a 3090 loader, the hold naming the 3090) is no head for one
  # it does not stop (an `any` loader, which takes the A6000), so that one is served at once while the held-back ticket keeps its name.
  qreset
  bash "$self" --test-locks "$t" --sitting-hold up sit 3090 $$ > /dev/null 2>&1
  pgate 0 H 2 3090
  p1=$PG
  ack 0 'sitting hold over the queue'
  s1=$ACK
  tk0=$(ls "$t/v41-queue.d")
  pgate 1 Q 2 any
  p2=$PG
  ack 1 'sitting hold over the queue'
  s2=$ACK
  tk1=$(ls "$t/v41-queue.d")
  rm -f "$t/sitting.gpuhold"
  grant 0 'sitting hold over the queue'
  s3=$ACK
  [ "$s2" = exit ] || grant 1 'sitting hold over the queue'
  rc=0
  wait "$p1" || rc=$?
  wait "$p2" || rc=$?
  echo "signs: $s1 $s2 $s3" > "$t/out"
  echo "held-back ticket while the other loader ran: $([ "$tk1" = "$tk0" ] && [ -n "$tk0" ] && echo same || echo "moved ($tk0 / $tk1)")" >> "$t/out"
  markers >> "$t/out"
  judge_all "the sitting hold: a loader it stops is no head for one it does not stop" "$rc" 0 \
    'signs: polled exit exit$' 'held-back ticket while the other loader ran: same$' 'markers: Q H $'
  pace_close 0; pace_close 1
  # A loader ended by TERM takes its ticket with it (the trap runs; its last sign says so).
  qreset
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  pgate 0 T 2 any
  p1=$PG
  ack 0 'TERM'
  kill -TERM "$(sed -n 's/.* pid=\([0-9]*\) .*/\1/p' "$t"/v41-queue.d/2-*)"
  ack 0 'TERM'
  wait "$p1" || true
  flock -u 6
  exec 6>&-
  echo "sign: $ACK tickets left: $(ls "$t/v41-queue.d" | wc -l | tr -d ' ')" > "$t/out"
  judge_all 'a queued loader ended by TERM removes its ticket' 0 0 'sign: exit tickets left: 0$'
  pace_close 0
  # A loader the bound ends (75) takes its ticket with it.
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  case_ 'a V4.1 loader that is not served within the bound is 75 and leaves no ticket' 75 'V4\.1 load lock .* was not free within 6 s' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_V41_CLASS=3
  flock -u 6
  exec 6>&-
  echo "tickets left: $(ls "$t/v41-queue.d" 2> /dev/null | wc -l | tr -d ' ')" > "$t/out"
  judge_all '  … the ticket is gone' 0 0 'tickets left: 0$'
  # A queue nobody waits in prints nothing, and the view is a read: box.sh under BLOOMERY_BOX_READONLY=1 runs it in the remote dir as it
  # is and syncs nothing (stub ssh and rsync that record their arguments and reach nothing).
  qreset
  rc=0
  bash "$self" --test-locks "$t" --v41-queue > "$t/out" 2>&1 || rc=$?
  judge_all '--v41-queue with nobody queued prints nothing and is 0' "$rc" 0 '^$'
  mkdir -p "$t/boxtree/tools" "$t/stub"
  cp "$(dirname "$self")/box.sh" "$t/boxtree/tools/"
  cp "$(dirname "$self")/../Cargo.toml" "$t/boxtree/"
  : > "$t/ssh.log"
  : > "$t/rsync.log"
  printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\n' "$t/ssh.log" > "$t/stub/ssh"
  printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\n' "$t/rsync.log" > "$t/stub/rsync"
  chmod +x "$t/stub/ssh" "$t/stub/rsync"
  rc=0
  BLOOMERY_BOX_LOCAL=1 env -u BLOOMERY_HOLD_OWNER -u BLOOMERY_BOX_WAIT -u BLOOMERY_BOX_ENV -u BLOOMERY_CARD -u BLOOMERY_ARCH -u BLOOMERY_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_V41_MODEL \
    -u BLOOMERY_DATA -u BLOOMERY_TIER -u BLOOMERY_FIXTURE_VARIANT -u BLOOMERY_BOX_LOCAL \
    PATH="$t/stub:$PATH" BLOOMERY_BOX=stub BLOOMERY_REMOTE='~/repo/x' BLOOMERY_BOX_READONLY=1 \
    bash "$t/boxtree/tools/box.sh" 'bash tools/gpu-gate.sh --v41-queue' > "$t/out" 2>&1 || rc=$?
  { echo "rsync calls: $(wc -l < "$t/rsync.log" | tr -d ' ')"; echo "last ssh: $(tail -1 "$t/ssh.log" | sed 's/.*&& //')"; } >> "$t/out"
  judge_all '--v41-queue runs under BLOOMERY_BOX_READONLY=1 and syncs nothing' "$rc" 0 'rsync calls: 0$' 'last ssh: bash tools/gpu-gate\.sh --v41-queue$'
  case_ 'BLOOMERY_V41_CLASS other than 1 to 4: 64, named' 64 "BLOOMERY_V41_CLASS is 1 .* got '5'" 3090 "$t/lease" BLOOMERY_V41_CLASS=5
  case_ '  … a name is no class either' 64 "BLOOMERY_V41_CLASS is 1 .* got 'two'" 3090 "$t/lease" BLOOMERY_V41_CLASS=two
  fi
  if gg_part 2; then
  # The run-private directory (EXE_LINK). A stub gate (gate_relink) says it is running and holds; the test then does what a concurrent build
  # does to the shared name (removes target/release/gate_relink and links another file there); the stub then opens the path it was
  # started as, which must still be its own image (marker gen1), as the CUDA anchor loader's re-open of its mapped image must.
  rl=$t/tree/target/release
  relink_stub() {
    printf '%s\n' '#!/bin/sh' '# gen1' '[ -z "${STUB_READY:-}" ] || echo "$0" > "$STUB_READY"' '[ -z "${STUB_HOLD:-}" ] || read -r _ < "$STUB_HOLD"' \
      'if grep -q "^# gen1\$" "$0"; then echo "image still $0"; else echo "image replaced: $0"; exit 3; fi' > "$rl/gate_relink"
    chmod +x "$rl/gate_relink"
  }
  printf '%s\n' '#!/bin/sh' '# gen2' 'exit 0' > "$t/relink-gen2"
  chmod +x "$t/relink-gen2"
  relink_stub
  mkfifo "$t/relink.hold"
  exec 19<> "$t/relink.hold"
  rm -f "$t/relink.ready"
  GATE_BIN=gate_relink gate "$t/relink.out" any "$t/lease" STUB_READY="$t/relink.ready" STUB_HOLD="$t/relink.hold" &
  rp=$!
  # The stub's first line is the event; the wait goes on while the run lives (the failsafe bound ends a hung one).
  while [ ! -s "$t/relink.ready" ] && kill -0 "$rp" 2> /dev/null; do sleep 0.1; done
  if [ -s "$t/relink.ready" ]; then
    rm -f "$rl/gate_relink"
    ln "$t/relink-gen2" "$rl/gate_relink"
    echo go >&19
  fi
  rc=0
  wait "$rp" || rc=$?
  exec 19>&-
  cp "$t/relink.out" "$t/out"
  judge_all 'a shared binary re-linked mid-run: the gate still runs from its own image' "$rc" 0 \
    '^image still \./target/release/\.run\.[0-9]+/gate_relink$' '!image replaced'
  seen=$(cat "$t/relink.ready" 2> /dev/null)
  { echo "ran as: $seen"; if [ -e "$t/tree/${seen%/*}" ]; then echo 'link: present'; else echo 'link: gone'; fi; echo "shared: $(grep -c '^# gen2$' "$rl/gate_relink")"; } > "$t/out"
  judge_all '  … the private directory is gone when the run ends, the shared name is left as the build made it' 0 0 \
    '^ran as: \./target/release/\.run\.[0-9]+/gate_relink$' '^link: gone$' '^shared: 1$'
  # --executable: the copy is made beside the executable, whatever directory that is.
  relink_stub
  gate_args=(./target/release/gate_relink)
  rc=0
  GATE_BIN=--executable gate "$t/out" any "$t/lease" || rc=$?
  gate_args=()
  judge_all '  … a run of --executable PATH takes the private directory beside the executable too' "$rc" 0 '^image still \./target/release/\.run\.[0-9]+/gate_relink$'
  # The siblings a gate spawns (generate_qwen3moe, bloomery-quantize, bloomery-serve) are found from the gate's own directory: with_file_name
  # for a bin (gate_sib), parent().parent() for a test artifact in deps/ (deps/gate_sib_t-0123). The sibling (sib_relink) is started, says
  # it is running and holds; the test re-links the shared sibling as a build does; the sibling then opens the path it was started as. FAIL-first:
  # before the private directory held the siblings the sibling started from the shared name and read "image replaced" (the CUDA anchor
  # loader's "No such file or directory").
  sib_stub() {
    rm -f "$rl/sib_relink" "$rl/gate_sib" "$rl/deps/gate_sib_t-0123"
    printf '%s\n' '#!/bin/sh' '# gen1' '[ -z "${STUB_READY:-}" ] || echo "$0" > "$STUB_READY"' '[ -z "${STUB_HOLD:-}" ] || read -r _ < "$STUB_HOLD"' \
      'if grep -q "^# gen1\$" "$0"; then echo "sibling image still $0"; else echo "sibling image replaced: $0"; exit 3; fi' > "$rl/sib_relink"
    printf '%s\n' '#!/bin/sh' 'exec "$(dirname "$0")/sib_relink"' > "$rl/gate_sib"
    mkdir -p "$rl/deps"
    printf '%s\n' '#!/bin/sh' 'exec "$(dirname "$0")/../sib_relink"' > "$rl/deps/gate_sib_t-0123"
    chmod +x "$rl/sib_relink" "$rl/gate_sib" "$rl/deps/gate_sib_t-0123"
  }
  sib_race() { # <GATE_BIN word> <label> <ERE>…: one run with the sibling held, the shared sibling re-linked, the sibling released
    local gbin=$1 label=$2
    shift 2
    sib_stub
    rm -f "$t/relink-gen2"
    printf '%s\n' '#!/bin/sh' '# gen2' 'exit 0' > "$t/relink-gen2"
    chmod +x "$t/relink-gen2"
    rm -f "$t/relink.ready"
    exec 19<> "$t/relink.hold"
    GATE_BIN=$gbin gate "$t/relink.out" any "$t/lease" STUB_READY="$t/relink.ready" STUB_HOLD="$t/relink.hold" &
    rp=$!
    while [ ! -s "$t/relink.ready" ] && kill -0 "$rp" 2> /dev/null; do sleep 0.1; done
    if [ -s "$t/relink.ready" ]; then
      rm -f "$rl/sib_relink"
      ln "$t/relink-gen2" "$rl/sib_relink"
      echo go >&19
    fi
    rc=0
    wait "$rp" || rc=$?
    exec 19>&-
    cp "$t/relink.out" "$t/out"
    judge_all "$label" "$rc" 0 "$@"
  }
  gate_args=()
  sib_race gate_sib 'a sibling the gate spawns, its shared name re-linked while it runs: it still runs from its own image' \
    '^sibling image still (\./|/.*/)target/release/\.run\.[0-9]+/sib_relink$' '!sibling image replaced'
  gate_args=(./target/release/deps/gate_sib_t-0123)
  sib_race --executable '  … the same for a test artifact in deps/, whose sibling is found from the directory above deps/' \
    '^sibling image still (\./|/.*/)target/release/\.run\.[0-9]+/deps/\.\./sib_relink$' '!sibling image replaced'
  gate_args=()
  ls -A "$rl" "$rl/deps" | sed 's/^/ls: /' > "$t/out"
  judge_all '  … both runs end with no private directory left' 0 0 '!^ls: \.run\.'
  rm -rf "$rl/deps" "$rl/sib_relink" "$rl/gate_sib"
  # A killed run leaves its directory (and the previous scheme left `.<NAME>.run.<pid>` links): the next run removes those whose pid is gone,
  # any name; a live pid's stay.
  dead=$(sh -c 'echo $$')
  me=$$
  mkdir -p "$rl/.run.$dead" "$rl/.run.$me"
  : > "$rl/.gate_relink.run.$dead"
  : > "$rl/.gate_other.run.$me"
  rc=0
  GATE_BIN=gate_relink gate "$t/out" any "$t/lease" || rc=$?
  ls -A "$rl" | sed 's/^/ls: /' >> "$t/out"
  judge_all 'a private directory or link whose pid is gone is removed by the next run; a live pid'"'"'s stays' "$rc" 0 \
    "removed the stale run-private \\./target/release/\\.run\\.$dead \\(pid $dead is gone\\)" \
    "removed the stale run-private \\./target/release/\\.gate_relink\\.run\\.$dead \\(pid $dead is gone\\)" \
    "!^ls: \\.run\\.$dead\$" "!^ls: \\.gate_relink\\.run\\.$dead\$" "^ls: \\.run\\.$me\$" "^ls: \\.gate_other\\.run\\.$me\$"
  rm -rf "$rl/.run.$me" "$rl/.gate_other.run.$me"
  # A directory that cannot be made is a named error and its own exit (73); the gate does not run from the shared path.
  mkdir -p "$t/lnfail"
  printf '%s\n' '#!/bin/sh' 'echo "ln: stub: Operation not permitted" >&2' 'exit 1' > "$t/lnfail/ln"
  chmod +x "$t/lnfail/ln"
  rc=0
  GATE_BIN=gate_relink gate "$t/out" any "$t/lease" PATH="$t/lnfail:$PATH" || rc=$?
  judge_all 'a private directory that cannot be made: 73, named, the gate does not run from the shared path' "$rc" 73 \
    'gate_relink: cannot make its run-private directory \./target/release/\.run\.[0-9]+ .*Operation not permitted' '!image still'
  fi
  if gg_part 3; then
  case_ 'BLOOMERY_V41_PRED_S not whole seconds: 64, named' 64 "BLOOMERY_V41_PRED_S is whole seconds .* got '2m'" 3090 "$t/lease" BLOOMERY_V41_PRED_S=2m
  # The tier. A fixture load takes no V4.1 load lock — only when its model is the file box.sh resolved: the lock file is not opened
  # (a directory stands there, which a real load refuses with 69), a held lock is not waited for, a request without the resolved
  # file is refused nothing and takes the lock as a real load does.
  rm -f "$t/v41-load.lock"
  mkdir "$t/v41-load.lock"
  FX='BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_MODEL=/m/fx-00001-of-00001.gguf'
  case_ 'a real load whose lock file cannot be opened: 69, named' 69 'cannot open .*v41-load\.lock' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  qreset
  # shellcheck disable=SC2086 # FX is K=V words
  case_ 'a fixture load opens no load lock and takes none' 0 'ok on the 3090 \(asked 3090\), tier fixture: no V4\.1 load lock' 3090 "$t/lease" \
    BLOOMERY_GATE_V41_LOAD=1 $FX BLOOMERY_REF_MODEL=/m/fx-00001-of-00001.gguf
  echo "queue dir: $([ -e "$t/v41-queue.d" ] && echo made || echo none)" > "$t/out"
  judge_all '  … and takes no ticket' 0 0 'queue dir: none$'
  rmdir "$t/v41-load.lock"
  : > "$t/v41-load.lock"
  exec 6> "$t/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the V4.1 load lock"; return 1; }
  rc=0
  # shellcheck disable=SC2086
  gate "$t/out" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 $FX BLOOMERY_REF_MODEL=/m/fx-00001-of-00001.gguf || rc=$?
  judge 'a fixture load does not wait for a held load lock' "$rc" 0 'ok on the 3090 \(asked 3090\), tier fixture: no V4\.1 load lock' 'waits for the V4\.1'
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
  # appends the file as it stands after (all three take a third argument, a directory: the hold of another --test-locks dir).
  fi
  hold() {
    printf 'owner=%s since=%s\n' "$1" "$(($(date +%s) - ${2:-0}))" > "${3:-$t}/batch.gpuhold"
    python3 -c 'import os, sys, time; a = time.time() - float(sys.argv[2]); os.utime(sys.argv[1], (a, a))' "${3:-$t}/batch.gpuhold" "${2:-0}"
  }
  hold_age() { python3 -c 'import os, sys, time; print(int(time.time() - os.stat(sys.argv[1]).st_mtime))' "$1"; }
  # hold_fresh <file> <t0> <t1>: yes when the file's mtime lies between the two epochs the test read around the writer's run — written by
  # that run, however late the test's own processes start — else no.
  hold_fresh() { python3 -c 'import os, sys; print("yes" if int(sys.argv[2]) <= int(os.stat(sys.argv[1]).st_mtime) <= int(sys.argv[3]) else "no")' "$@"; }
  hrun() {
    local t0 t1
    rc=0
    t0=$(date +%s)
    bash "$self" --test-locks "$t" --gpuhold "$@" > "$t/out" 2>&1 || rc=$?
    t1=$(date +%s)
    if [ -e "$t/batch.gpuhold" ]; then
      printf 'after: %s age=%s fresh=%s\n' "$(head -1 "$t/batch.gpuhold")" "$(hold_age "$t/batch.gpuhold")" "$(hold_fresh "$t/batch.gpuhold" "$t0" "$t1")" >> "$t/out"
    else
      echo 'after: no hold' >> "$t/out"
    fi
  }
  if gg_part 3; then
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
  # This hold is released by the test; use the helper cap while bound checks retain their own limits.
  TL="$t/d3" gate "$t/h6" 3090 "$t/lease" GATE_TEST_CAP="$cap" &
  p6=$!
  # And a stale hold over a 3090 lock that stays taken: the run polls for the card for the whole bound and names the stale hold once.
  mkdir -p "$t/d4"
  hold other 400 "$t/d4"
  exec 5> "$t/d4/gate.lock"
  flock -x 5 || { echo "FAIL: the test cannot hold the d4 gate lock"; return 1; }
  TL="$t/d4" gate "$t/h7" 3090 "$t/lease" &
  p7=$!
  # d3's hold is released once its run is waiting on it (its first [batch-hold] line) and has polled once more — an event of the run's
  # own clock, not a fixed time after the launch: the others wait out their 4 s bound beside it. The run is waited for while it lives.
  k=$SECONDS
  while ! grep -q 'batch-hold' "$t/h6" 2> /dev/null && kill -0 "$p6" 2> /dev/null && [ $((SECONDS - k)) -lt "$fs" ]; do
    sleep 0.2
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
  # PIN(2026-10-19): the waited seconds were `[4-9]`. The runner prints the wait it measured, never under the 4 s bound it was
  # given and longer on a loaded host, so a ceiling of 9 s read the load: the floor stays, and the line must name the bound
  # it was given (CARD_BOUND 4 s). The same two bands below.
  judge_all 'a fresh hold of another owner: the run waits, 75 at the bound, a [batch-hold] line at once naming owner and age' "$rc" 75 \
    '\[batch-hold\] .* waits: a landing batch holds the GPUs \(.*batch\.gpuhold, owner other, up [0-9]+ s, refreshed [0-9]+ s ago\)' \
    'the batch hold .*batch\.gpuhold \(owner other\) was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\) .*contention' '!ok on the'
  rc=0
  wait "$p2" || rc=$?
  cp "$t/h2" "$t/out"
  judge_all '  … an owner that is a prefix of the hold'"'"'s is not the owner' "$rc" 75 '\[batch-hold\] .* owner other' '!ok on the'
  rc=0
  wait "$p3" || rc=$?
  cp "$t/h3" "$t/out"
  judge_all '  … a V4.1 loader waits on the hold before its load lock: the hold'"'"'s bound, not the load lock'"'"'s' "$rc" 75 \
    'the batch hold .* was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\)' '!V4\.1 load lock .* was not free'
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
  # PIN(2026-10-19): the age was `4[0-9][0-9]`. The hold is written 400 s old and its age grows by the time the case takes to
  # reach it, which a loaded host can stretch past 499 s: the floor stays (past the 300 s a hold goes stale at), the ceiling
  # is 999. The same band for the takeover below.
  judge_all 'a stale hold (not refreshed for 400 s): the run goes ahead, and says so once, naming the hold and its age' "$rc" 0 \
    'ok on the 3090 \(asked 3090\)' '!waits' \
    '\[batch-hold\] .*batch\.gpuhold \(owner other\) is stale — not refreshed for [4-9][0-9][0-9] s \(stale at 300 s\)' 'hold lines: 1$'
  case_ 'BLOOMERY_BATCH_OWNER with a character other than letters, digits, _: 64, named' 64 "BLOOMERY_BATCH_OWNER is the batch hold's owner .*got 'a-b'" 3090 "$t/lease" BLOOMERY_BATCH_OWNER=a-b
  # A round batch's item (BLOOMERY_HOLD_YIELD=1, the header) does not wait the hold out: the first probe that finds it up ends the run 75,
  # naming the owner, inside the 4 s a waiting run spends on the same hold. A run that passes today still passes with it set: the hold's own
  # owner, a stale hold, no hold.
  hold other 0
  y0=$SECONDS
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_HOLD_YIELD=1 || rc=$?
  echo "yielded inside the bound: $([ $((SECONDS - y0)) -lt 4 ] && echo yes || echo no)" >> "$t/out"
  judge_all 'a round item meets a fresh hold: it yields at the first probe, naming the owner, without waiting' "$rc" 75 \
    'the batch hold .*batch\.gpuhold \(owner other\) is up: yielding at the first probe .*contention' 'yielded inside the bound: yes$' \
    '!was still up after' '!waits: a landing batch' '!ok on the'
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_HOLD_YIELD=1 BLOOMERY_BATCH_OWNER=other || rc=$?
  judge_all "  … the hold's own batch still passes with it set" "$rc" 0 'ok on the 3090 \(asked 3090\)' '!batch-hold|batch hold'
  hold other 400
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_HOLD_YIELD=1 || rc=$?
  judge_all '  … a stale hold is ignored as before' "$rc" 0 'ok on the 3090 \(asked 3090\)' '!yielding'
  rm -f "$t/batch.gpuhold"
  rc=0
  gate "$t/out" 3090 "$t/lease" BLOOMERY_HOLD_YIELD=1 || rc=$?
  judge_all '  … no hold, no yield' "$rc" 0 'ok on the 3090 \(asked 3090\)' '!yielding|batch hold'
  case_ 'BLOOMERY_HOLD_YIELD other than 1 or 0: 64, named' 64 "BLOOMERY_HOLD_YIELD is 1 .*got '2'" 3090 "$t/lease" BLOOMERY_HOLD_YIELD=2
  # The writer: up, beat and down on the file, owner-checked; the wait above reads what it writes.
  hrun up batcha
  judge_all 'gpuhold up: puts the hold up with its owner and the box clock' "$rc" 0 'up: owner batcha$' 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=yes$'
  hrun up batcha
  judge_all '  … again by the same owner: refreshed, not refused' "$rc" 0 'up: owner batcha$' 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=yes$'
  hrun up other
  judge_all "  … another owner's fresh hold is refused, 75, and stays" "$rc" 75 'is up for owner batcha .*not taking it \(rc 75\)' 'after: owner=batcha ' '!owner=other'
  hold batcha 200
  hrun beat batcha
  judge_all 'gpuhold beat: refreshes the mtime of its own hold' "$rc" 0 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=yes$'
  hold batcha 200
  hrun beat other
  judge_all "  … never another owner's" "$rc" 3 "owner batcha's, not other's: not refreshing it \(rc 3\)" 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=no$'
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
  judge_all 'gpuhold up: takes over a stale hold, naming it' "$rc" 0 'owner other was not refreshed for [4-9][0-9][0-9] s: taking it over' 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=yes$'
  rm -f "$t/batch.gpuhold"
  hrun up 'a b'
  judge_all 'gpuhold: an owner with a space: 64, named' "$rc" 64 "an owner is letters, digits and _, got 'a b'" 'after: no hold$'
  # The file's name stays apart from the sittings' holds (/root/bloomery-<owner>-hold): builds and every other box command ignore it.
  out=$(bash "$self" --gpuhold path)
  out="$out $(env -u BLOOMERY_LEASE_HOLDS bash -c '. "$1"; case /root/bloomery-03-hold in $LEASE_HOLDS) printf "sitting-hold-matches " ;; esac; case "$2" in $LEASE_HOLDS) echo MATCHES ;; *) echo apart ;; esac' _ \
    "$(dirname "$self")/ref/lease-probe.sh" "$out")"
  printf '%s\n' "$out" > "$t/out"
  judge_all "the batch hold's name does not match the sittings' LEASE_HOLDS" 0 0 '^/root/bloomery-batch\.gpuhold sitting-hold-matches apart$'
  # The sitting hold (the header): a file of its own, named for the cards a release sitting times, alive as long as its pid. $$ is a live
  # pid the cases point it at, the pid of a finished child a dead one. Each case that waits has a directory of its own, so its own hold.
  rm -f "$t/batch.gpuhold" "$t/sitting.gpuhold"
  srun() { # <dir> <verb> …: the writer, its output and the file after it in $t/out
    local d=$1
    shift
    rc=0
    bash "$self" --test-locks "$d" --sitting-hold "$@" > "$t/out" 2>&1 || rc=$?
    if [ -e "$d/sitting.gpuhold" ]; then printf 'after: %s\n' "$(head -1 "$d/sitting.gpuhold")" >> "$t/out"; else echo 'after: no hold' >> "$t/out"; fi
  }
  sleep 0 &
  dpid=$!
  wait "$dpid"
  mkdir -p "$t/sh1" "$t/sh2" "$t/sh3" "$t/sh4" "$t/sh5" "$t/sh6" "$t/sh7" "$t/sh9"
  srun "$t/sh7" up sit 3090 $$
  judge_all 'sitting hold up: puts it up with its owner, cards, pid and start' "$rc" 0 'sitting hold .*sitting\.gpuhold up: owner sit cards 3090$' \
    "after: owner=sit cards=3090 pid=$$ start=[^ ]+ since=[0-9]+\$"
  srun "$t/sh7" up other 3090,a6000 $$
  judge_all "  … a live hold is not taken, whoever asks: 75, and it stays" "$rc" 75 'sitting\.gpuhold is up for owner sit \(pid [0-9]+\): not taking it \(rc 75\)' 'after: owner=sit cards=3090 ' '!owner=other'
  srun "$t/sh7" down other
  judge_all "sitting hold down: leaves another owner's" "$rc" 0 "is owner sit's, not other's: left up" 'after: owner=sit '
  srun "$t/sh7" down sit
  judge_all '  … removes its own' "$rc" 0 'sitting hold .*down: owner sit$' 'after: no hold$'
  srun "$t/sh7" down sit
  judge_all '  … and is no error when it is already down' "$rc" 0 'is already down$' 'after: no hold$'
  printf 'owner=gone cards=3090 pid=%s start=Mon1 since=1\n' "$dpid" > "$t/sh7/sitting.gpuhold"
  srun "$t/sh7" up sit2 a6000 $$
  judge_all 'sitting hold up: takes over a dead hold (its pid is gone), naming it' "$rc" 0 'owner gone \(pid [0-9]+\) is dead: taking it over' 'after: owner=sit2 cards=a6000 '
  srun "$t/sh7" down sit2
  srun "$t/sh7" up sit 5090 $$
  judge_all 'sitting hold: cards that are none of 3090, a6000, 3090,a6000, local: 64, named' "$rc" 64 "the cards are 3090, a6000, 3090,a6000 or local, got '5090'" 'after: no hold$'
  srun "$t/sh7" up sit local $$
  judge_all '  … local is a hold of its own: a sitting on a machine in local mode puts it up' "$rc" 0 'sitting hold .*sitting\.gpuhold up: owner sit cards local$' "after: owner=sit cards=local pid=$$ start=[^ ]+ since=[0-9]+\$"
  srun "$t/sh7" down sit
  srun "$t/sh7" up sit 3090 "$dpid"
  judge_all '  … a pid that is not a live process: 64, named' "$rc" 64 "pid $dpid is not a live process" 'after: no hold$'
  srun "$t/sh7" up 'a b' 3090 $$
  judge_all '  … an owner with a space: 64, named' "$rc" 64 "an owner is letters, digits and _, got 'a b'" 'after: no hold$'
  out=$(bash "$self" --sitting-hold path)
  out="$out $(env -u BLOOMERY_LEASE_HOLDS bash -c '. "$1"; case "$2" in $LEASE_HOLDS) echo MATCHES ;; *) echo apart ;; esac' _ "$(dirname "$self")/ref/lease-probe.sh" "$out")"
  printf '%s\n' "$out" > "$t/out"
  judge_all "the sitting hold's name is neither the batch hold's nor a sitting's LEASE_HOLDS" 0 0 '^/root/bloomery-sitting\.gpuhold apart$'
  # Waiting: a hold on the 3090 stops a 3090 run (75 at the 4 s bound, a [sitting-hold] line at once naming owner, cards and pid) and lets the
  # A6000 run go; a hold on the A6000 sends an `any` run to the 3090; a hold on both stops `any` and `both`, and a V4.1 loader before its
  # load lock; a hold with nothing readable is up for both; a dead hold is ignored, named once; a hold released while the run waits.
  srun "$t/sh1" up sit 3090 $$
  srun "$t/sh2" up sit a6000 $$
  srun "$t/sh3" up sit 3090,a6000 $$
  : > "$t/sh4/sitting.gpuhold"
  printf 'owner=gone cards=3090,a6000 pid=%s start=Mon1 since=1\n' "$dpid" > "$t/sh5/sitting.gpuhold"
  srun "$t/sh6" up sit 3090 $$
  srun "$t/sh9" up sit local $$
  exec 6> "$t/sh3/v41-load.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the s3 V4.1 load lock"; return 1; }
  TL="$t/sh1" gate "$t/sg1" 3090 "$t/lease" &
  p1=$!
  TL="$t/sh1" gate "$t/sg2" a6000 "$t/lease" &
  p2=$!
  TL="$t/sh2" gate "$t/sg3" any "$t/lease" &
  p3=$!
  TL="$t/sh3" gate "$t/sg4" any "$t/lease" &
  p4=$!
  TL="$t/sh3" gate "$t/sg5" both "$t/lease" BLOOMERY_BOX_CARD=both &
  p5=$!
  TL="$t/sh3" gate "$t/sg6" 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1 &
  p6=$!
  TL="$t/sh4" gate "$t/sg7" 3090 "$t/lease" &
  p7=$!
  TL="$t/sh5" gate "$t/sg8" 3090 "$t/lease" &
  p8=$!
  TL="$t/sh6" gate "$t/sg9" 3090 "$t/lease" GATE_TEST_CAP="$cap" &
  p9=$!
  TL="$t/sh1" gate "$t/sg10" any "$t/lease" &
  p10=$!
  TL="$t/sh9" gate "$t/sg12" a6000 "$t/lease" &
  p12=$!
  TL="$t/sh9" gate "$t/sg13" any "$t/lease" &
  p13=$!
  mkdir -p "$t/sh8"
  srun "$t/sh8" up sit 3090 $$
  exec 5> "$t/sh8/gate-a6000.lock"
  flock -x 5 || { echo "FAIL: the test cannot hold the sh8 A6000 gate lock"; return 1; }
  TL="$t/sh8" gate "$t/sg11" any "$t/lease" &
  p11=$!
  k=$SECONDS
  while ! grep -q 'sitting-hold' "$t/sg9" 2> /dev/null && kill -0 "$p9" 2> /dev/null && [ $((SECONDS - k)) -lt "$fs" ]; do
    sleep 0.2
  done
  sleep 1
  prc3=0 prca=0
  flock -n "$t/sh3/gate.lock" true || prc3=$?
  flock -n "$t/sh3/gate-a6000.lock" true || prca=$?
  echo "probe 3090 lock rc=$prc3 a6000 lock rc=$prca" > "$t/out"
  judge_all 'runs waiting on the sitting hold hold no card lock' 0 0 'probe 3090 lock rc=0 a6000 lock rc=0$'
  rm -f "$t/sh6/sitting.gpuhold"
  rc=0
  wait "$p1" || rc=$?
  cp "$t/sg1" "$t/out"
  judge_all 'a sitting hold on the 3090: a 3090 run waits, 75 at the bound, a [sitting-hold] line at once naming owner, cards and pid' "$rc" 75 \
    '\[sitting-hold\] .* waits: a release sitting holds the cards \(.*sitting\.gpuhold, owner sit, cards 3090, pid [0-9]+, up [0-9]+ s\)' \
    'the sitting hold .*sitting\.gpuhold \(owner sit, cards 3090\) was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\) .*contention' '!ok on the'
  rc=0
  wait "$p2" || rc=$?
  cp "$t/sg2" "$t/out"
  judge_all '  … an A6000 run goes ahead, no hold line' "$rc" 0 'ok on the a6000 \(asked a6000\)' '!sitting-hold|sitting hold'
  rc=0
  wait "$p10" || rc=$?
  cp "$t/sg10" "$t/out"
  judge_all '  … an `any` run takes the A6000' "$rc" 0 'ok on the a6000 \(asked any\)' '!sitting-hold|sitting hold'
  rc=0
  wait "$p11" || rc=$?
  flock -u 5
  exec 5>&-
  cp "$t/sg11" "$t/out"
  judge_all '  … with the A6000 busy an `any` run waits for it, never taking the held 3090' "$rc" 75 'no gate lock \(any\) was free within 4 s' '!ok on'
  rc=0
  wait "$p12" || rc=$?
  cp "$t/sg12" "$t/out"
  judge_all 'a sitting hold on local is a hold on every card: an A6000 run waits, 75 at the bound' "$rc" 75 '\[sitting-hold\] .* owner sit, cards local, pid' \
    'the sitting hold .*sitting\.gpuhold \(owner sit, cards local\) was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\)' '!ok on'
  rc=0
  wait "$p13" || rc=$?
  cp "$t/sg13" "$t/out"
  judge_all '  … and so does an `any` run, which no single held card turns away' "$rc" 75 '\[sitting-hold\] .* cards local' '!ok on'
  rc=0
  wait "$p3" || rc=$?
  cp "$t/sg3" "$t/out"
  judge_all 'a sitting hold on the A6000: an `any` run, which looks at the A6000 first, takes the 3090' "$rc" 0 'ok on the 3090 \(asked any\)' '!sitting-hold|sitting hold'
  rc=0
  wait "$p4" || rc=$?
  cp "$t/sg4" "$t/out"
  judge_all 'a sitting hold on both: an `any` run waits' "$rc" 75 '\[sitting-hold\] .* cards 3090,a6000' 'the sitting hold .* was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\)' '!ok on'
  rc=0
  wait "$p5" || rc=$?
  cp "$t/sg5" "$t/out"
  judge_all '  … a run of both cards waits' "$rc" 75 '\[sitting-hold\] .* owner sit' '!ok on'
  rc=0
  wait "$p6" || rc=$?
  flock -u 6
  exec 6>&-
  cp "$t/sg6" "$t/out"
  judge_all "  … a V4.1 loader waits on the hold before its load lock: the hold's bound, not the load lock's" "$rc" 75 \
    'the sitting hold .* was still up after ([4-9]|[1-9][0-9]+) s \(CARD_BOUND 4 s\)' '!V4\.1 load lock .* was not free'
  rc=0
  wait "$p7" || rc=$?
  cp "$t/sg7" "$t/out"
  judge_all 'a hold with nothing readable is up for both cards, never free' "$rc" 75 '\[sitting-hold\] .* owner \?, cards 3090,a6000' '!ok on'
  rc=0
  wait "$p8" || rc=$?
  cp "$t/sg8" "$t/out"
  echo "hold lines: $(grep -c 'sitting-hold' "$t/out")" >> "$t/out"
  judge_all 'a dead hold (its pid is gone): the run goes ahead and says so once, naming the hold, its owner and pid' "$rc" 0 \
    'ok on the 3090 \(asked 3090\)' '\[sitting-hold\] .*sitting\.gpuhold \(owner gone, pid [0-9]+\) is dead — its sitting is gone; ignoring it' 'hold lines: 1$' '!waits'
  rc=0
  wait "$p9" || rc=$?
  cp "$t/sg9" "$t/out"
  judge_all 'a hold released while the run waits: it goes ahead and counts the wait on its own line' "$rc" 0 \
    'ok on the 3090 \(asked 3090\)' '^gpu-gate\.sh: waited [0-9]+ s for the sitting hold$'
  # box.sh's local mode (BLOOMERY_BOX_CARD=local, its lock directory the test's): one lock for this machine's card(s), apart from
  # the box's two, and the caller's CUDA_VISIBLE_DEVICES (the stub prints it); a held lease is waited for as the lock is.
  fi
  local ON_LOCAL="ok on this machine's card\\(s\\), the local gate lock" LOC=(BLOOMERY_BOX_CARD=local "BLOOMERY_LOCK_DIR=$t")
  if gg_part 3; then
  rc=0
  # PIN(2026-11-17): the caller's pin is a card the stub lists (the 3090's UUID, which an `any` lane pick would not choose first); it was
  # the made-up GPU-caller before gpu-gate.sh refused a pin that names no card (the two cases below).
  gate "$t/out" any "$t/lease" "${LOC[@]}" CUDA_VISIBLE_DEVICES="$GPU_3090" || rc=$?
  judge_all 'local: any takes the local lock and leaves CUDA_VISIBLE_DEVICES as it was' "$rc" 0 "$ON_LOCAL \\(asked any\\)$" "^ran on $GPU_3090\$"
  case_ '  … a pin that names no card of the machine: 64, named, and no lock opened' 64 'CUDA_VISIBLE_DEVICES=GPU-gone, set by its environment .*names no card this machine has .*nvidia-smi lists: ' any "$t/lease" "${LOC[@]}" CUDA_VISIBLE_DEVICES=GPU-gone
  case_ '  … the same pin on a lane run (3090): 64, named' 64 'CUDA_VISIBLE_DEVICES=GPU-gone, set by its environment .*names no card this machine has' 3090 "$t/lease" CUDA_VISIBLE_DEVICES=GPU-gone
  case_ '  … a6000 takes it too' 0 "$ON_LOCAL \\(asked a6000\\)$" a6000 "$t/lease" "${LOC[@]}"
  cp "$t/tree/target/release/ok" "$t/tree/target/release/gate_qwen38_serve"
  gate_args=(--arm plain)
  GATE_BIN=gate_qwen38_serve case_ "  … the Qwen3.8 serve gate's plain arm runs on another local card in the real tier" 0 "^ran on $GPU_3090\$" any "$t/lease" "${LOC[@]}" CUDA_VISIBLE_DEVICES="$GPU_3090" BLOOMERY_TIER=real
  gate_args=()
  case_ '  … a V4.1 load takes the V4.1 load lock with it' 0 "$ON_LOCAL \\(asked any\\), V4\\.1 load lock$" any "$t/lease" "${LOC[@]}" BLOOMERY_GATE_V41_LOAD=1
  exec 5> "$t/gate.lock" 6> "$t/gate-a6000.lock"
  { flock -x 5 && flock -x 6; } || { echo "FAIL: the test cannot hold the two card locks"; return 1; }
  case_ "  … the box's two card locks held: it runs" 0 "$ON_LOCAL \\(asked any\\)$" any "$t/lease" "${LOC[@]}"
  exec 5>&- 6>&-
  exec 6> "$t/gate-local.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the local lock"; return 1; }
  case_ '  … the local lock held: 75 at the bound' 75 'no gate lock \(local\) was free within 4 s' any "$t/lease" "${LOC[@]}"
  exec 6>&-
  # On the box local mode's one lock is the box's gate lock, so the box's other runs hold it off.
  : > "$t/box-lease.lock"
  exec 5> "$t/gate.lock"
  flock -x 5 || { echo "FAIL: the test cannot hold the gate lock"; return 1; }
  case_ '  … on the box, the gate lock held: 75 at the bound' 75 'no gate lock \(local\) was free within 4 s' any "$t/lease" "${LOC[@]}"
  exec 5>&-
  rm -f "$t/box-lease.lock"
  exec 7> "$t/lease"
  flock -x 7 || { echo "FAIL: the test cannot hold its lease file"; return 1; }
  case_ '  … the lease held: it times the card, 75 at the bound' 75 'no gate lock \(local\) was free within 4 s' any "$t/lease" "${LOC[@]}"
  fi
  exec 7>&-
  if gg_part 4; then
  # Only the compiler and hardware discovery are replaced. Both entry points below use the
  # real runner, flock, timeout and subprocess cleanup against the same isolated lock files.
  mkdir -p "$t/tree/package/native"
  printf '%s\n' package-data > "$t/tree/package/consumer.txt"
  # Build a real loader consumer: Python may link math statically, so it is not a shared-object fixture.
  python3 - "$t/tree/package/native/libgate-fixture.so" <<'PY' || return 1
import subprocess
import sys

if sys.platform == "linux":
    subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-o", sys.argv[1], "-"],
                   input="int gate_fixture_value(void) { return 37; }\n", text=True, check=True)
PY
  printf '%s\n' '#!/bin/sh' 'printf "host: %s\n" "${LIB_HOST:-x86_64-unknown-linux-gnu}"' > "$t/bin/rustc"
  cat > "$t/bin/cargo" <<'PY'
#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

args = sys.argv[1:]
if args[0] == "oxide":
    args = args[args.index("--") + 1:]
test_args = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
exe = os.environ.get("LIB_EXE", os.path.abspath("target/release/lib-test"))
package = os.path.abspath("package")
context = dict(os.environ, CARGO_MANIFEST_DIR=package)
context["LD_LIBRARY_PATH"] = package + "/native" + (":" + context["LD_LIBRARY_PATH"] if context.get("LD_LIBRARY_PATH") else "")
if os.environ.get("LIB_ARGV"):
    Path(os.environ["LIB_ARGV"]).write_text("".join(arg + "\n" for arg in sys.argv[1:]), encoding="utf-8")
print("compiler output before capture", flush=True)
time.sleep(float(os.environ.get("LIB_COMPILE_SLEEP", "0")))
if os.environ.get("LIB_COMPILED"):
    Path(os.environ["LIB_COMPILED"]).write_text("compiled\n", encoding="utf-8")
code = int(os.environ.get("LIB_COMPILE_EXIT", "0"))
if code:
    sys.exit(code)
if "--no-run" in args:
    sys.exit(0)
if "--config" not in args:
    os.chdir(package)
    os.execve(exe, [exe, *test_args], context)
# Cargo merges every --config; the runner is the one that sets target.<triple>.runner.
config = next(c for c in (tomllib.loads(args[i + 1]) for i, arg in enumerate(args) if arg == "--config") if "target" in c)
# Cargo looks the runner up under the triple it builds for: --target's, else the host's.
target = args[args.index("--target") + 1] if "--target" in args else os.environ.get("LIB_HOST", "x86_64-unknown-linux-gnu")
runner = [config["target"][target]["runner"]]
for _ in range(int(os.environ.get("LIB_CAPTURES", "1"))):
    subprocess.run([*runner, exe, *test_args], cwd=package, env=context, check=True)
if os.environ.get("LIB_CAPTURE_CORRUPT"):
    next(Path(runner[0]).parent.glob("run-*.json")).write_text("{broken", encoding="utf-8")
print("compiler output after capture", flush=True)
PY
  cat > "$t/tree/target/release/lib-test" <<'PY'
#!/usr/bin/env python3
import ctypes
import os
import signal
import sys
from pathlib import Path

# Consumer behavior: package-relative data, Cargo's manifest environment, and the native loader
# must still work after replay; echoing the captured environment would not prove this.
assert Path("consumer.txt").read_text(encoding="utf-8") == "package-data\n"
assert (Path(os.environ["CARGO_MANIFEST_DIR"]) / "consumer.txt").read_text(encoding="utf-8") == "package-data\n"
if sys.platform == "linux":
    assert ctypes.CDLL("libgate-fixture.so").gate_fixture_value() == 37

if "hw_held_" in sys.argv and os.environ.get("RUST_TEST_THREADS") != "1":
    sys.exit(97)
with open(os.environ["LIB_STARTED"], "w", encoding="utf-8") as stream:
    stream.write(str(os.getpid()))
# GNU timeout sends its TERM to the child and again to its group: os._exit ends at once, where sys.exit's shutdown puts the
# default action back and a late second TERM would end the test with 143 in place of LIB_TERM.
signal.signal(signal.SIGTERM, lambda *_: os._exit(int(os.environ.get("LIB_TERM", "143"))))
if os.environ.get("LIB_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.environ.get("LIB_PID"):
    with open(os.environ["LIB_PID"], "w", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()} {os.getppid()}\n")
if os.environ.get("LIB_READY"):
    with open(os.environ["LIB_READY"], "w", encoding="utf-8") as stream:
        stream.write("up\n")
if os.environ.get("LIB_HOLD"):
    with open(os.environ["LIB_HOLD"], encoding="utf-8") as stream:
        stream.readline()
print("lib-card=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("test result: ok. " + os.environ.get("LIB_PASSED", "1") + " passed; 0 failed;")
sys.exit(int(os.environ.get("LIB_EXIT", "0")))
PY
  chmod +x "$t/bin/cargo" "$t/bin/rustc" "$t/tree/target/release/lib-test"
  local lib_build=() lib_target=(--lib) # Cargo build options and the test target
  lib_gate() {
    local o=$1 c=$2
    shift 2
    (cd "$t/tree" && timeout --kill-after=10 "$((cap + 30))" "${gate_env[@]}" \
      BLOOMERY_GATE_CARD="$c" BLOOMERY_LEASE_LOCK="$t/lease" BLOOMERY_LEASE_PROC="$t/proc" \
      LIB_COMPILED="$t/compiled" LIB_STARTED="$t/started" RUSTC="$t/bin/rustc" CARGO_HOME="$t/cargo-home" "$@" \
      bash -c 'echo "$$" > "$1"; shift; exec bash "$@"' _ "$t/lib-runner" \
      "${self%/*}/gate.sh" --test-locks "${TL:-$t}" --gpu --oxide -p bloomery-gpu --release ${lib_build[@]+"${lib_build[@]}"} "${lib_target[@]}" -- --include-ignored "${LIB_FILTER:-probe}") > "$o" 2>&1
  }
  # Binary libtests share the same capture, card and model-load locks as library tests.
  for selector in --bin --bin=lib-test; do
    lib_target=("$selector")
    [ "$selector" != --bin ] || lib_target+=(lib-test)
    rc=0
    lib_gate "$t/out" 3090 BLOOMERY_GATE_V41_LOAD=1 CUDA_VISIBLE_DEVICES="$GPU_3090" LIB_STARTED="$t/bin-success-started" || rc=$?
    judge_all "binary test $selector replays with both locks" "$rc" 0 'V4\.1 load lock' "^lib-card=$GPU_3090$"
    exec 5> "$t/gate.lock"
    flock -x 5 || return 1
    rc=0
    lib_gate "$t/out" 3090 LIB_STARTED="$t/bin-card-started" || rc=$?
    [ -e "$t/bin-card-started" ] || echo child-not-started >> "$t/out"
    judge_all "binary test $selector waits for the shared card lock" "$rc" 75 'no gate lock' '^child-not-started$'
    exec 5>&-
    exec 7> "$t/v41-load.lock"
    flock -x 7 || return 1
    rc=0
    lib_gate "$t/out" 3090 BLOOMERY_GATE_V41_LOAD=1 LIB_STARTED="$t/bin-load-started" || rc=$?
    [ -e "$t/bin-load-started" ] || echo child-not-started >> "$t/out"
    judge_all "binary test $selector waits for the shared model-load lock" "$rc" 75 'V4\.1 load lock.*not free' '^child-not-started$'
    exec 7>&-
    for captures in 0 2; do
      rc=0
      lib_gate "$t/out" 3090 LIB_CAPTURES="$captures" LIB_STARTED="$t/bin-refused-started" || rc=$?
      [ -e "$t/bin-refused-started" ] || echo child-not-started >> "$t/out"
      judge_all "binary test $selector refuses $captures captured executables" "$rc" 64 "select exactly one test target; got $captures" '^child-not-started$'
    done
  done
  lib_target=(--lib)
  # Run the checkpoint input recipe itself with the compiler and box transport stubbed.
  mkdir -p "$t/tree/tools/ref"
  printf '%s\n' '#!/usr/bin/env bash' 'exec bash -c "$1"' > "$t/tree/tools/box.sh"
  printf '#!/usr/bin/env bash\nif [ "${1:-}" = --gpu ]; then exec bash %q --test-locks %q "$@"; fi\nexec bash %q "$@"\n' \
    "${self%/*}/gate.sh" "$t" "${self%/*}/gate.sh" > "$t/tree/tools/gate.sh"
  ln -s "${self%/*}/ref/needs-env.sh" "$t/tree/tools/ref/needs-env.sh"
  chmod +x "$t/tree/tools/box.sh"
  exec 7> "$t/v41-load.lock"
  flock -x 7 || return 1
  for tier in real fixture; do
    rc=0
    (cd "$t/tree" && "${gate_env[@]}" BLOOMERY_GATE_CARD=3090 BLOOMERY_LEASE_LOCK="$t/lease" \
      BLOOMERY_LEASE_PROC="$t/proc" BLOOMERY_QWEN38_FOLDER="$t/checkpoint folder" LIB_STARTED="$t/recipe-$tier-started" \
      BLOOMERY_TIER="$tier" BLOOMERY_REF_MODEL="$t/fixture.gguf" BLOOMERY_FIXTURE_MODEL="$t/fixture.gguf" \
      RUSTC="$t/bin/rustc" CARGO_HOME="$t/cargo-home" \
      just --justfile "${self%/*}/../justfile" --working-directory "$t/tree" arch=sm_86 gate-gpu-qwen4exp-mtp-input) > "$t/out" 2>&1 || rc=$?
    [ -e "$t/recipe-$tier-started" ] || echo child-not-started >> "$t/out"
    judge_all "checkpoint input recipe in $tier tier waits for the shared model-load lock" "$rc" 75 'V4\.1 load lock.*not free' '^child-not-started$'
  done
  exec 7>&-
  # Ordinary binary holds the card: compilation completes, but libtest never starts.
  fifos
  gate "$t/first" 3090 "$t/lease" CUDA_VISIBLE_DEVICES="$GPU_3090" STUB_READY="$t/ready" STUB_HOLD="$t/hold" &
  p1=$!
  event 'ordinary gate holds the card'
  rc=0
  lib_gate "$t/out" 3090 || rc=$?
  [ -f "$t/compiled" ] && echo compiled >> "$t/out"
  [ ! -e "$t/started" ] && echo child-not-started >> "$t/out"
  judge_all 'lib test contends with an ordinary GPU bin, but compiles outside its lock' "$rc" 75 '^compiled$' '^child-not-started$'
  rc=0
  lib_gate "$t/out" a6000 || rc=$?
  judge 'lib test can use the other card while the ordinary bin holds the 3090' "$rc" 0 "^lib-card=$GPU_A6000$"
  rc=0
  lib_gate "$t/out" any || rc=$?
  judge 'lib test any selects the free A6000 while the 3090 is held' "$rc" 0 "^lib-card=$GPU_A6000$"
  rc=0
  lib_gate "$t/out" any BLOOMERY_BOX_CARD=local BLOOMERY_LOCK_DIR="$t" CUDA_VISIBLE_DEVICES="$GPU_3090" || rc=$?
  judge 'lib test local mode preserves the caller visible devices' "$rc" 0 "^lib-card=$GPU_3090\$"
  rc=0
  (cd "$t/tree" && "${gate_env[@]}" LIB_STARTED="$t/host-started" \
    bash "${self%/*}/gate.sh" --release -p bloomery-verbs --lib) > "$t/out" 2>&1 || rc=$?
  [ -f "$t/host-started" ] && echo host-started >> "$t/out"
  judge 'host-only verbs path runs without taking the held card lock' "$rc" 0 '^host-started$'
  rm -f "$t/started"
  streams "$t/lib-output"
  lib_gate "$t/lib-output" 3090 CUDA_VISIBLE_DEVICES="$GPU_3090" GATE_TEST_CAP="$cap" &
  p2=$!
  relay "$t/lib-output" "$t/out" 'waits for the gate lock' 3 &
  r1=$!
  event 'lib test queues behind ordinary bin'
  local child_waited=no
  [ -e "$t/started" ] || child_waited=yes
  echo go >&4
  rc=0 rc1=0
  wait "$p2" || rc=$?
  wait "$p1" || rc1=$?
  wait "$r1"
  cat "$t/first" >> "$t/out"
  [ "$child_waited" != yes ] || echo child-not-started >> "$t/out"
  judge_all 'lib test starts after the ordinary bin releases the shared card' "$rc" 0 '^child-not-started$' "^lib-card=$GPU_3090$"
  judge 'ordinary holder exits successfully on the same card' "$rc1" 0 "^ran on $GPU_3090$"
  # Reverse direction, including the held pass's real serial-env requirement.
  fifos
  LIB_FILTER=hw_held_ lib_gate "$t/lib-first" 3090 RUST_TEST_THREADS=1 LIB_READY="$t/ready" LIB_HOLD="$t/hold" &
  p1=$!
  event 'held lib pass owns the card'
  case_ 'ordinary GPU bin cannot start while the held lib pass owns its card' 75 'no gate lock' 3090 "$t/lease"
  echo go >&4
  rc=0
  wait "$p1" || rc=$?
  cp "$t/lib-first" "$t/out"
  judge 'held lib pass finishes under its serial environment' "$rc" 0 '^lib-card='
  case_ 'ordinary GPU bin runs after the held lib pass releases its card' 0 '^ran on ' 3090 "$t/lease"
  for row in 'LIB_EXIT=23:23' 'LIB_PASSED=0:78' 'LIB_COMPILE_EXIT=29:29' 'LIB_CAPTURES=0:64' 'LIB_CAPTURES=2:64' 'LIB_CAPTURE_CORRUPT=1:1' "LIB_EXE=$t/tree:2" "LIB_EXE=$t/missing:2" 'BLOOMERY_GATE_BOUND=0:64' 'BLOOMERY_GATE_CARD=bad:64'; do
    rm -f "$t/started"
    rc=0
    lib_gate "$t/out" 3090 "${row%:*}" || rc=$?
    judge "lib path preserves ${row%:*}" "$rc" "${row##*:}" '.|^$'
    case "$row" in
      LIB_COMPILE_EXIT=* | LIB_CAPTURE* | LIB_EXE=* | BLOOMERY_*)
        [ ! -e "$t/started" ] && echo child-not-started >> "$t/out"
        judge 'compile failure or refused input starts no test child' "$rc" "${row##*:}" '^child-not-started$' ;;
    esac
  done
  # The build is the host build every other oxide recipe makes: no --target (it would key the whole dependency cone apart from
  # theirs). A cross triple that cargo would still take from the call, the environment or a config is refused by name before cargo runs.
  rm -f "$t/argv"
  rc=0
  lib_gate "$t/out" 3090 LIB_ARGV="$t/argv" || rc=$?
  { [ -f "$t/argv" ] && sed 's/^/cargo-arg /' "$t/argv"; } >> "$t/out" || echo no-cargo-call >> "$t/out"
  judge_all 'lib path builds for the host: the cargo call passes no --target' "$rc" 0 '^cargo-arg --config$' '!^cargo-arg --target' '!^no-cargo-call$'
  local NOTARGET="names the target aarch64-unknown-linux-gnu, not the host x86_64-unknown-linux-gnu"
  mkdir -p "$t/tree/.cargo"
  printf '%s\n' '[build]' 'target = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]' > "$t/tree/.cargo/config.toml"
  for refusal in "build-config-file:\\[build\\] target of .*/\\.cargo/config\\.toml $NOTARGET" "CARGO_BUILD_TARGET:CARGO_BUILD_TARGET $NOTARGET" \
      "config-flag:--config build\\.target $NOTARGET" "config-flag-pair:--config build\\.target $NOTARGET"; do
    rm -f "$t/compiled" "$t/started"
    lib_build=()
    rc=0
    case "${refusal%%:*}" in
      build-config-file) lib_gate "$t/out" 3090 || rc=$? ;;
      CARGO_BUILD_TARGET) lib_gate "$t/out" 3090 CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu || rc=$? ;;
      config-flag) lib_build=(--config=build.target=aarch64-unknown-linux-gnu); lib_gate "$t/out" 3090 || rc=$? ;;
      config-flag-pair) lib_build=(--config 'build.target=["aarch64-unknown-linux-gnu"]'); lib_gate "$t/out" 3090 || rc=$? ;;
    esac
    [ ! -e "$t/compiled" ] && echo cargo-not-run >> "$t/out"
    [ ! -e "$t/started" ] && echo child-not-started >> "$t/out"
    judge_all "lib path refuses a cross target from ${refusal%%:*}" "$rc" 64 "${refusal#*:}" '^cargo-not-run$' '^child-not-started$'
  done
  lib_build=()
  # The same triple as the host is no cross target; cargo's own order (--config, the environment, the nearest config file) decides which source counts.
  rc=0
  lib_gate "$t/out" 3090 CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu || rc=$?
  judge 'lib path accepts the host triple as CARGO_BUILD_TARGET' "$rc" 0 '^lib-card='
  printf '%s\n' '[build]' 'target = "aarch64-unknown-linux-gnu"' > "$t/tree/.cargo/config.toml"
  rc=0
  lib_gate "$t/out" 3090 CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu || rc=$?
  judge 'lib path: the environment outranks a cross [build] target' "$rc" 0 '^lib-card='
  lib_build=(--config 'build.target="x86_64-unknown-linux-gnu"')
  rc=0
  lib_gate "$t/out" 3090 || rc=$?
  lib_build=()
  judge 'lib path: --config outranks a cross [build] target' "$rc" 0 '^lib-card='
  rm -rf "$t/tree/.cargo"
  mkdir -p "$t/lib-unopenable/gate.lock"
  rc=0
  TL="$t/lib-unopenable" lib_gate "$t/out" 3090 || rc=$?
  judge 'lib path keeps lock-open failure code' "$rc" 69 '.'
  # The capture call is the build (tools/gate.sh's header): it has a bound of its own, and the card test the run bound. A compile slower
  # than the run bound but inside the build bound goes through to the test; one past the build bound ends 124 naming the build phase
  # and starts no test. The compile of the first case outlasts the run bound (5 s) by 3 s.
  rm -f "$t/started"
  rc=0
  lib_gate "$t/out" 3090 BLOOMERY_GATE_BOUND=5 BLOOMERY_GATE_BUILD_BOUND=60 LIB_COMPILE_SLEEP=8 || rc=$?
  judge 'lib path: a build slower than the run bound, inside the build bound, runs the test' "$rc" 0 '^lib-card='
  rm -f "$t/started"
  rc=0
  lib_gate "$t/out" 3090 BLOOMERY_GATE_BOUND=60 BLOOMERY_GATE_BUILD_BOUND=2 LIB_COMPILE_SLEEP=20 || rc=$?
  [ ! -e "$t/started" ] && echo child-not-started >> "$t/out"
  judge_all 'lib path: a build past the build bound ends 124 naming the build phase and starts no test' "$rc" 124 \
    'GATE TIMED OUT in the build phase after the 2s build bound' '^child-not-started$'
  # A runtime bound and an external signal both stop the child and release the lock.
  for stop in bound kill signal; do
    fifos
    rm -f "$t/started"
    if [ "$stop" = bound ] || [ "$stop" = kill ]; then
      lib_gate "$t/lib-first" 3090 BLOOMERY_GATE_BOUND=1 LIB_HOLD="$t/hold" LIB_IGNORE_TERM="$([ "$stop" = kill ] && echo 1)" &
    else
      rm -f "$t/lib.pid"
      lib_gate "$t/lib-first" 3090 LIB_PID="$t/lib.pid" LIB_READY="$t/ready" LIB_HOLD="$t/hold" LIB_TERM=23 &
    fi
    p1=$!
    if [ "$stop" = signal ]; then
      # The ready line, then the test's monitor asleep past its fork (the held run's case above says why), is the event.
      event 'lib test ready for cancellation'
      since=$SECONDS
      until cancel_ready "$t/lib.pid" || [ $((SECONDS - since)) -ge "$fs" ]; do sleep 0.05; done
      kill -TERM "$(cat "$t/lib-runner")"
    fi
    rc=0
    wait "$p1" || rc=$?
    cp "$t/lib-first" "$t/out"
    case "$stop" in bound) prc=124 ;; kill) prc=137 ;; signal) prc=23 ;; esac
    judge "lib path propagates $stop" "$rc" "$prc" '.'
    if [ -f "$t/started" ]; then
      pid_start "$(cat "$t/started")"
      [ -n "$PID_START" ] || echo child-stopped >> "$t/out"
    fi
    judge "lib child is stopped after $stop" "$rc" "$prc" '^child-stopped$'
    case_ "shared card lock is free after lib $stop" 0 '^ran on ' 3090 "$t/lease"
  done
  fi
  selftest_load_down
  echo "gpu-gate self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}
# --self-test: every part in a process of its own, their lines in part order, one summary of the counts. The guards in this file
# must name exactly parts 1..SELFTEST_PARTS: a part with no guard would run no case, a guard past the last part would never run.
self_test_parts() {
  local self tmp k line a b ok=0 total=0 red=0 guards want='' slowest
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  guards=$(grep -o 'gg_part [0-9][0-9]*; then' "$self" | grep -o '[0-9][0-9]*' | sort -un | tr '\n' ' ')
  for ((k = 1; k <= SELFTEST_PARTS; k++)); do want="$want$k "; done
  if [ "$guards" != "$want" ]; then
    echo "gpu-gate.sh: self-test: the guards name parts ${guards:-none}, not 1..$SELFTEST_PARTS" >&2
    return 70
  fi
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/gpu-gate-parts.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "selftest_load_down; rm -rf '$tmp'" EXIT
  selftest_load_up || return $?
  for ((k = 1; k <= SELFTEST_PARTS; k++)); do
    # The load runs once, in this parent. A part that also saw the variable would start another set.
    ( set +e; s0=$SECONDS; env -u GPU_GATE_SELFTEST_LOAD bash "$self" --self-test --part "$k" > "$tmp/$k.out" 2>&1; echo $? > "$tmp/$k.rc"; echo $((SECONDS - s0)) > "$tmp/$k.s" ) &
  done
  wait
  selftest_load_down
  for ((k = 1; k <= SELFTEST_PARTS; k++)); do
    cat "$tmp/$k.out"
    line=$(tail -n 1 "$tmp/$k.out")
    if [[ $line =~ ^gpu-gate\ self-test:\ ([0-9]+)\ of\ ([0-9]+)\ ok$ ]]; then
      a=${BASH_REMATCH[1]} b=${BASH_REMATCH[2]}
      ok=$((ok + a)) total=$((total + b))
      [ "$a" = "$b" ] || red=1
    else
      red=1
      echo "gpu-gate self-test: part $k ended without its count line"
    fi
    [ "$(cat "$tmp/$k.rc" 2> /dev/null)" = 0 ] || { red=1; echo "gpu-gate self-test: part $k ended rc $(cat "$tmp/$k.rc" 2> /dev/null || echo none)"; }
  done
  slowest=$(for ((k = 1; k <= SELFTEST_PARTS; k++)); do echo "$(cat "$tmp/$k.s" 2> /dev/null || echo 0) part $k"; done | sort -rn | awk 'NR <= 3 {printf "%s%s %s %s s", (NR > 1 ? ", " : ""), $2, $3, $1}')
  echo "gpu-gate self-test: slowest parts: $slowest"
  echo "gpu-gate self-test: $ok of $total ok"
  [ "$red" = 0 ]
}
if [ "${1:-}" = --self-test ]; then
  case $# in
    1) self_test_parts; exit $? ;;
    3) [ "$2" = --part ] && { [ "$3" = all ] || { [ "$3" -ge 1 ] 2> /dev/null && [ "$3" -le "$SELFTEST_PARTS" ]; }; } || { echo "gpu-gate.sh: --self-test --part takes all or 1..$SELFTEST_PARTS" >&2; exit 64; }
      self_test "$3"; exit $? ;;
    *) echo "gpu-gate.sh: --self-test takes nothing else, or --part K|all" >&2; exit 64 ;;
  esac
fi
if [ "${1:-}" = --test-locks ]; then
  [ $# -ge 3 ] && [ -d "$2" ] || { echo "usage: gpu-gate.sh --test-locks <dir> <binary> [args...] (the self-test's)" >&2; exit 64; }
  GATE_LOCK=$2/gate.lock A6000_LOCK=$2/gate-a6000.lock LOCAL_LOCK=$2/gate-local.lock V41_LOCK=$2/v41-load.lock GPU_HOLD=$2/batch.gpuhold SIT_HOLD=$2/sitting.gpuhold V41_BOUND=6 POLL=1 CARD_BOUND=4
  V41_QDIR=$2/v41-queue.d V41_SEQ=$2/v41-queue.seq
  # The box's mark is the test's too, so a case reads the machine the test makes, not the one it runs on (the box has /root's mark).
  BOX_LEASE=$2/box-lease.lock
  # A helper waiter, one whose life the test ends by an event (it releases the lock the waiter queues on), is set up with a safety cap
  # in place of the two bounds; the cases that judge a bound never set it.
  if [ -n "${GATE_TEST_CAP:-}" ]; then
    case $GATE_TEST_CAP in 0 | *[!0-9]*) echo "gpu-gate.sh: GATE_TEST_CAP is whole seconds from 1, got '$GATE_TEST_CAP'" >&2; exit 64 ;; esac
    V41_BOUND=$GATE_TEST_CAP CARD_BOUND=$GATE_TEST_CAP
  fi
  # GATE_TEST_PACE=<FIFO prefix>: a loader whose polls the test paces (the pace function); only here is the variable read.
  PACE=${GATE_TEST_PACE:-}
  shift 2
# On the box (the header) every run but the self-test's takes the locks under /root.
# PIN(2026-11-17): BLOOMERY_BOX_CARD=local no longer ends here on the box. The box holds one card now and its environment file declares the cards in
# view (tools/box.sh); the refusal's reason, a run that holds none of the box's locks, is the lock directory, so local mode on the box is
# the box's own lock directory (below: the one lock is the gate lock the box's other runs take), and another directory is what is refused.
elif on_box && [ "$LOCK_DIR" != /root ]; then
  echo "gpu-gate.sh: BLOOMERY_LOCK_DIR=$LOCK_DIR on the machine whose lease is $BOX_LEASE (the box): its gates and sittings lock under /root, and this run would hold none of their locks; run it through tools/box.sh without local mode. If this machine is not the box, a lock script run as root without BLOOMERY_LOCK_DIR made that file: remove it" >&2
  exit 64
fi
if [ "${1:-}" = --gpuhold ]; then
  shift
  gpuhold "$@"
  exit $?
fi
if [ "${1:-}" = --sitting-hold ]; then
  shift
  sitting_hold "$@"
  exit $?
fi
if [ "${1:-}" = --v41-queue ]; then
  v41_queue_view
  exit 0
fi
DIRECT=0
if [ "${1:-}" = --executable ]; then
  DIRECT=1
  shift
fi
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: gpu-gate.sh <target/release binary> [args...] | --executable <path> [args...]" >&2; exit 64; }
shift
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/gate-bound.sh"
gate_bound gpu-gate.sh || exit $?
# shellcheck source=tools/pack-for-driver.sh
source "${BASH_SOURCE[0]%/*}/pack-for-driver.sh" || exit 70 # a failed source does not stop `set -uo pipefail`
if [ "$DIRECT" = 1 ]; then
  EXE=$NAME
  case $EXE in /* | ./* | ../*) ;; *) EXE=./$EXE ;; esac
  NAME=${EXE##*/}
else
  EXE=./target/release/$NAME
fi
[ -f "$EXE" ] && [ -x "$EXE" ] || { echo "gpu-gate.sh: no executable file $EXE — the recipe builds it before calling this" >&2; exit 2; }
# BLOOMERY_BUILD_ONLY=1 (tools/gate-batch.sh's build lane puts it in an item's box env; tools/gate.sh reads the same variable): the
# recipe's build ran before this call, and the binary is there. The run ends here, 0, before any lock, queue ticket or card: it runs
# nothing and holds nothing. 0 or unset runs the gate as always; any other value ends 64.
case ${BLOOMERY_BUILD_ONLY:-0} in
  0) ;;
  1) echo "gpu-gate.sh: BLOOMERY_BUILD_ONLY: $EXE is built; nothing runs, no lock is taken"; exit 0 ;;
  *) echo "gpu-gate.sh: BLOOMERY_BUILD_ONLY is 1 (build, run nothing) or 0/unset, got '$BLOOMERY_BUILD_ONLY'" >&2; exit 64 ;;
esac
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
# BLOOMERY_HOLD_YIELD: `1` yields to a batch hold at the first probe instead of waiting for it (the header).
case ${BLOOMERY_HOLD_YIELD:-0} in
  0 | 1) ;;
  *) echo "gpu-gate.sh: BLOOMERY_HOLD_YIELD is 1 (yield to a batch hold at the first probe) or 0/unset, got '$BLOOMERY_HOLD_YIELD'" >&2; exit 64 ;;
esac
# BLOOMERY_V41_CLASS and BLOOMERY_V41_PRED_S: the V4.1 load queue's class and the calling recipe's predicted wall (the header).
CLASS=${BLOOMERY_V41_CLASS:-2}
case $CLASS in
  1 | 2 | 3 | 4) ;;
  *) echo "gpu-gate.sh: BLOOMERY_V41_CLASS is 1 (the lead's landing batch), 2 (round gates, the default), 3 (sittings) or 4 (idle fill), got '$CLASS'" >&2; exit 64 ;;
esac
PRED=${BLOOMERY_V41_PRED_S:--}
case $PRED in
  -) ;;
  '' | *[!0-9]*) echo "gpu-gate.sh: BLOOMERY_V41_PRED_S is whole seconds (the recipe's median wall) or unset, got '$PRED'" >&2; exit 64 ;;
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
# V4.1 `--place gate` 게이트는 any 없이 3090에 고정해 돌리니(gate-batch.sh의 레인 A) 떠도는 게이트가 3090을 비워
# 두는 편이 낫다. any는 잡힌 타이밍 임대가 재는 카드(아래 timing_card)를 건너뛰고, A6000은 컴퓨트 프로세스가
# 있어도 건너뛴다. 카드를 강제한 실행은 그 카드가 잡힌 임대의 타이밍 카드면 75로 끝난다. 임대 탐침은
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
  local)
    # a machine that is not the box and names /root would make the file that marks the box; the box itself (its lease file is there) is
    # the one machine whose local run locks under /root
    if box_lock_dir "$LOCK_DIR" && ! on_box; then
      echo "gpu-gate.sh: box.sh's local mode (BLOOMERY_BOX_CARD=local) keeps its lock under BLOOMERY_LOCK_DIR, and that is $LOCK_DIR (unset: /root), the box's lock directory: run it through tools/box.sh's local mode, which sets both" >&2
      exit 64
    fi
    ASK=${BLOOMERY_GATE_CARD:-3090}
    case $ASK in
      3090 | a6000 | any) CARD=local ;;
      both)
        # shellcheck source=tools/ref/cards.sh
        BLOOMERY_BOX_CARD=local . "${BASH_SOURCE[0]%/*}/ref/cards.sh"
        cards_view && [ "$CARDS_VIEW_N" -ge 2 ] || {
          echo "gpu-gate.sh: BLOOMERY_GATE_CARD=both runs on two or more cards in view, and ${CARDS_VIEW_WHY:-this machine has $CARDS_VIEW_N in view: $CARDS_VIEW}" >&2
          exit 64
        }
        CARD=local ;;
      *) echo "gpu-gate.sh: box.sh's local mode (BLOOMERY_BOX_CARD=local) runs on the cards in view under one lock, and BLOOMERY_GATE_CARD=$ASK names no card: leave it unset" >&2; exit 64 ;;
    esac ;;
  *) echo "gpu-gate.sh: BLOOMERY_BOX_CARD is 3090, a6000, both or local (box.sh sets it), got '$BOXC'" >&2; exit 64 ;;
esac
[ "$BOXC" = local ] || case "$CARD" in
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
  local u processes; u=$(uuid_of A6000); [ -n "$u" ] || return 1
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || return 1
  ! grep -Fq -- "$u" <<< "$processes"
}
# 락 파일을 못 여는 셸(박스의 root가 아니거나 박스가 아님)은 경쟁이 아니다 — 여기서 이름을 대고 끝난다.
LOCK_FILES=("9:$GATE_LOCK" "8:$A6000_LOCK")
# On the box the one lock of the cards in view is the gate lock the box's other runs, an older copy of this script on another track, the
# dump scripts and the release sitting already take for its card: a lock of another name would leave them off the same card unexcluded.
[ "$CARD" != local ] || { on_box && LOCAL_LOCK=$GATE_LOCK; LOCK_FILES=("9:$LOCAL_LOCK"); }
# A card pin that names no card this machine has ends the run here by name, before any lock is taken (tools/ref/cards.sh): the binary
# would otherwise see an empty device list and fall to the host engine. BLOOMERY_BUILD_ONLY ended above, before any card.
{ BLOOMERY_BOX_CARD=local . "${BASH_SOURCE[0]%/*}/ref/cards.sh" && cards_pin_check gpu-gate.sh "its environment (tools/box.sh's refusal names the file it sourced)"; } || exit $?
# The pack phase (the header) is here, after every refusal above and before the first lock is opened: no card is held while it packs.
pack_for_driver gpu-gate.sh "$EXE" || exit $?
[ "$V41" = 0 ] || LOCK_FILES+=("7:$V41_LOCK")
for lock in "${LOCK_FILES[@]}"; do
  eval "exec ${lock%%:*}>\"\${lock#*:}\"" 2> /dev/null || { echo "gpu-gate.sh: cannot open ${lock#*:} — the gate runs on the box, as root$([ "$CARD" != local ] || echo ", or in box.sh's local mode with a BLOOMERY_LOCK_DIR it can write")" >&2; exit 69; }
done
# A V4.1 loader queues (the header) before its first poll, and its ticket goes with the run on every exit.
if [ "$V41" = 1 ]; then
  trap on_exit EXIT
  ticket_write "$@"
fi
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
  sit_card_held a6000 && return 1
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
  sit_card_held 3090 && return 1
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
# take_local: the local lock (box.sh's local mode). The lease, held or in doubt, times this machine's card: the run lets the lock
# go and polls on, as `any` passes a timing card by.
take_local() {
  flock -n 9 || return 1
  if ! lease_free; then
    flock -u 9
    return 1
  fi
  GOT=local
}
# take_card: one try, without blocking, at the card lock(s) this run asked for; GOT names what it holds, empty when none.
take_card() {
  GOT=
  case "$CARD" in
    3090) take_3090 || true ;;
    a6000) take_a6000 || true ;;
    any) take_a6000 || take_3090 || true ;;
    local) take_local || true ;;
  esac
}
# take_both: the 3090 lock first, then, holding it, the A6000's. Every other run holds one card lock and never waits while
# holding it, and a loader's head waits here holding only the load lock, which no card holder waits for (the header's lock order),
# so this order cannot deadlock; a blocked flock wakes at the release before the polls of one-card runs
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
    3090 | local) flock -u 9 ;;
    a6000) flock -u 8 ;;
    both) flock -u 9; flock -u 8 ;;
  esac
  GOT=
}
# The V4.1 load lock: one V4.1 load on the box at a time, across trees and batches — two at once push each other's
# host set (~190 GB of the 256 GB) out of the page cache and both turn IO-bound. It is held by the binary for the whole
# run: the descriptor stays open, so the lock lives exactly as long as the load's process.
# A loader is served by the queue (the header): each poll, a run that is not the head only re-reads the queue and touches no card, no
# nvidia-smi and no lock; the head takes the load lock without blocking and then, holding it, its card(s) — a card wait of the head is
# the one wait that holds the load lock, so a later arrival never passes it. A run that is not a loader tries its card alone.
# The waits are the bounds' own: CARD_BOUND for a poll that found no card, V41_BOUND for one that found the lock held or a place behind
# another loader, each 75 (contention, not a red gate) with the lock named; a queued loader says its place at once and once a minute, the
# head names the lock's holders too (the card wait: a line, no holders).
# Each poll's seconds go to one of the waits — the batch hold, the card lock, the load lock and the queue (both `waited ... load lock`) —
# so the `waited` lines never overlap: tools/gate-batch.sh subtracts them all.
# The batch hold (the header) is read first in every poll, before the queue and before any lock: a run under a hold holds nothing while
# it waits (a head lets the load lock go), and a gate already polling cannot slip in between a batch's items.
GOT=
CARD_W=0 LOAD_W=0 HOLD_W=0 SIT_W=0 why='' said=-60 hsaid=-60 ssaid=-60 csaid=-60 qsaid=-60 last=$SECONDS HAVE7=0 qnote=''
try_cards() {
  if [ "$CARD" = both ]; then take_both; else take_card; fi
}
while :; do
  why=
  if batch_hold_blocks; then
    why=hold
    if [ "$HAVE7" = 1 ]; then
      flock -u 7
      HAVE7=0
    fi
  elif sit_hold_blocks; then
    why=sitting
    if [ "$HAVE7" = 1 ]; then
      flock -u 7
      HAVE7=0
    fi
  elif [ "$V41" = 1 ]; then
    if [ "$HAVE7" = 0 ]; then
      v41_standing "${TICKET##*/}" || true
      if [ "$Q_POS" = 0 ]; then
        echo "gpu-gate.sh: $NAME: its V4.1 queue ticket $TICKET is gone (removed by hand, or a reader read this process as dead) — not queueing again (rc 75)" >&2
        exit 75
      elif [ "$Q_POS" != 1 ]; then
        why=queue
      elif flock -n 7; then
        HAVE7=1
      else
        why=load
      fi
    fi
    if [ -z "$why" ]; then
      try_cards
      [ -n "$GOT" ] || why=card
    fi
  else
    try_cards
    [ -n "$GOT" ] || why=card
  fi
  [ -n "$why" ] || break
  if [ "$why" = hold ] && [ "${BLOOMERY_HOLD_YIELD:-0}" = 1 ]; then
    echo "gpu-gate.sh: $NAME: the batch hold $GPU_HOLD (owner $HOLD_OWNER) is up: yielding at the first probe (BLOOMERY_HOLD_YIELD=1), not waiting for it — contention, not a red gate (rc 75)" >&2
    exit 75
  fi
  if [ "$why" = hold ] && [ $((HOLD_W - hsaid)) -ge 60 ]; then
    up_s='?'
    [ -z "$HOLD_SINCE" ] || up_s=$(($(date +%s) - HOLD_SINCE))
    echo "[batch-hold] $(now) $NAME waits: a landing batch holds the GPUs ($GPU_HOLD, owner $HOLD_OWNER, up $up_s s, refreshed $HOLD_AGE s ago), holding no lock; ${HOLD_W} s so far, at most ${CARD_BOUND} s with the lock waits" >&2
    hsaid=$HOLD_W
  fi
  if [ "$why" = sitting ] && [ $((SIT_W - ssaid)) -ge 60 ]; then
    up_s='?'
    [ -z "$SIT_SINCE" ] || up_s=$(($(date +%s) - SIT_SINCE))
    echo "[sitting-hold] $(now) $NAME waits: a release sitting holds the cards ($SIT_HOLD, owner $SIT_OWNER, cards $SIT_CARDS, pid $SIT_PID, up $up_s s), holding no lock; ${SIT_W} s so far, at most ${CARD_BOUND} s with the lock waits" >&2
    ssaid=$SIT_W
  fi
  if [ "$why" = card ] && [ $((CARD_W - csaid)) -ge 60 ]; then
    echo "gpu-gate.sh: $NAME waits for the gate lock ($CARD), holding $([ "$HAVE7" = 1 ] && echo 'the V4.1 load lock' || echo 'no lock'); ${CARD_W} s so far, at most ${CARD_BOUND} s with the other waits" >&2
    csaid=$CARD_W
  fi
  if { [ "$why" = queue ] || [ "$why" = load ]; } && [ $((LOAD_W - qsaid)) -ge 60 ]; then
    queued_say
    qsaid=$LOAD_W
  fi
  if [ "$why" = load ] && [ $((LOAD_W - said)) -ge 60 ]; then
    echo "gpu-gate.sh: $NAME waits for the V4.1 load lock $V41_LOCK (another run is loading V4.1), holding no gate lock; ${LOAD_W} s so far, its holders:" >&2
    lease_holders "$V41_LOCK" >&2
    said=$LOAD_W
  fi
  pace
  spent=$((SECONDS - last)) last=$SECONDS
  case $why in
    card) CARD_W=$((CARD_W + spent)) ;;
    hold) HOLD_W=$((HOLD_W + spent)) ;;
    sitting) SIT_W=$((SIT_W + spent)) ;;
    *) LOAD_W=$((LOAD_W + spent)) ;;
  esac
  # The hold's seconds and the card lock's share CARD_BOUND: a batch's hold is a wait for the cards, as the card lock is.
  if [ $((CARD_W + HOLD_W + SIT_W)) -ge "$CARD_BOUND" ] && [ "$why" = card ]; then
    echo "gpu-gate.sh: no gate lock ($CARD) was free within ${CARD_BOUND} s — contention, not a red gate" >&2
    exit 75
  fi
  if [ $((CARD_W + HOLD_W + SIT_W)) -ge "$CARD_BOUND" ] && [ "$why" = hold ]; then
    echo "gpu-gate.sh: $NAME: the batch hold $GPU_HOLD (owner $HOLD_OWNER) was still up after ${HOLD_W} s (CARD_BOUND ${CARD_BOUND} s) — contention, not a red gate (rc 75)" >&2
    exit 75
  fi
  if [ $((CARD_W + HOLD_W + SIT_W)) -ge "$CARD_BOUND" ] && [ "$why" = sitting ]; then
    echo "gpu-gate.sh: $NAME: the sitting hold $SIT_HOLD (owner $SIT_OWNER, cards $SIT_CARDS) was still up after ${SIT_W} s (CARD_BOUND ${CARD_BOUND} s) — contention, not a red gate (rc 75)" >&2
    exit 75
  fi
  if [ "$LOAD_W" -ge "$V41_BOUND" ] && { [ "$why" = load ] || [ "$why" = queue ]; }; then
    [ "$why" != queue ] || qnote="; queued at $Q_POS of $Q_N (head: $Q_HEAD_ITEM $Q_HEAD_OWNER)"
    echo "gpu-gate.sh: $NAME: the V4.1 load lock $V41_LOCK was not free within ${V41_BOUND} s — contention, not a red gate (rc 75)$qnote" >&2
    exit 75
  fi
done
# What the last poll spent (take_both's blocking) was a wait for the card lock.
CARD_W=$((CARD_W + SECONDS - last))
ticket_drop
# The waits are not the gate's time: tools/gate-batch.sh subtracts these lines' seconds from the item's times row.
[ "$HOLD_W" = 0 ] || echo "gpu-gate.sh: waited ${HOLD_W} s for the batch hold" >&2
[ "$SIT_W" = 0 ] || echo "gpu-gate.sh: waited ${SIT_W} s for the sitting hold" >&2
[ "$CARD_W" = 0 ] || echo "gpu-gate.sh: waited ${CARD_W} s for the gate lock ($CARD)" >&2
[ "$LOAD_W" = 0 ] || echo "gpu-gate.sh: waited ${LOAD_W} s for the V4.1 load lock" >&2
if [ "$GOT" = a6000 ] || [ "$CARD" = any ]; then
  U=$(uuid_of "$([ "$GOT" = a6000 ] && echo A6000 || echo 3090)")
  [ -n "$U" ] || { echo "gpu-gate.sh: the $GOT lookup failed" >&2; exit 75; }
  export CUDA_VISIBLE_DEVICES=$U
fi
case $GOT in
  both) ON='both cards, both gate locks' ;;
  local) ON="this machine's card(s), the local gate lock" ;;
  *) ON="the $GOT" ;;
esac
# Every lock is held: the binary's card-byte invariant refuses a reading past its plan (crates/gpu-gates/src/card_bytes.rs).
export BLOOMERY_CARD_HELD=1
exe_link
echo "gpu-gate.sh: $NAME on $ON (asked $([ "$CARD" = local ] && echo "$ASK" || echo "$CARD"))$([ "$V41" = 0 ] || echo ', V4.1 load lock')$TIER_NOTE" >&2
# The run's own wall, its locks' waits outside it, is the closing line: a recipe that runs several
# binaries, or one binary several times, is read per run.
RUN_T0=$SECONDS
if [ -n "$STACKS" ]; then
  watch_traps=$(trap -p TERM INT HUP)
  # shellcheck source=tools/ref/stack-watch.sh
  source "${BASH_SOURCE[0]%/*}/ref/stack-watch.sh" "$STACKS" "$NAME" -- timeout --kill-after=10 "$BOUND" "$EXE" "$@"
  watch_rc=$?
  trap - TERM INT HUP
  eval "$watch_traps"
  (exit "$watch_rc")
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
  run_bounded timeout --kill-after=10 "$BOUND" gdb -q -batch -x "$GDB_CMDS" --args "$EXE" "$@"
  rc=$?
  rm -f "$GDB_CMDS"
  (exit "$rc")
else
  run_bounded timeout --kill-after=10 "$BOUND" "$EXE" "$@"
fi
rc=$?
echo "gpu-gate.sh: $NAME ran $((SECONDS - RUN_T0)) s (exit $rc)" >&2
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "GPU GATE TIMED OUT: $NAME after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "GPU GATE RED: $NAME (exit $rc)" >&2
fi
exit "$rc"
