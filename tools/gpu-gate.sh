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
# flock과 timeout이 필요하다 — 맥에는 없다). 시험만 쓰는 첫 인자 `--test-locks DIR`이 세 락과 배치 홀드 파일(batch.gpuhold), V4.1 적재 큐(v41-queue.d, v41-queue.seq)를 DIR 아래로 옮기고
# V4.1 적재 락의 상한을 6초로, 카드 락의 상한을 4초로, 폴링을 1초로 줄인다. GATE_TEST_CAP=<초> (the self-test's helper waiters only): both bounds become that safety cap; GATE_TEST_PACE=<FIFO prefix> (the self-test's paced loaders only): each poll rests until the test grants the next one.
# 환경: BLOOMERY_GATE_BOUND(초, 기본 900 — tools/gate.sh와 같은 레버), BLOOMERY_GATE_CARD(아래 카드 고르기),
# BLOOMERY_BOX_CARD(box.sh가 넘기는 카드 선택 — 아래), BLOOMERY_GATE_V41_LOAD(1이면 V4.1 적재 락도 잡는다 — 아래).
# BLOOMERY_BATCH_OWNER(letters, digits, _: the owner of the batch hold this run's batch put up — the batch hold paragraph below).
# BLOOMERY_GATE_STACKS(초 — 설정하면 바이너리의 출력이 그만큼 멈출 때 tools/ref/stack-watch.sh가 스레드 스택을 뜨고 끝낸다).
# BLOOMERY_LOCK_DIR: the directory of every lock and hold file named below, /root when unset (tools/ref/lease-probe.sh's LOCK_DIR).
# BLOOMERY_BOX_CARD=local (tools/box.sh's local mode): the run takes one lock, bloomery-gate-local.lock, for the card(s) this
# machine shows when BLOOMERY_GATE_CARD is unset, 3090, a6000 or any (`both` and other names: 64), and leaves
# CUDA_VISIBLE_DEVICES as it finds it. A held timing lease is read as timing that card: the run waits as for its lock. Its
# lock directory is never /root (64).
# Where /root/bloomery-cpu.lock exists (the box, as lease_take reads it), BLOOMERY_BOX_CARD=local and a BLOOMERY_LOCK_DIR other
# than /root are refused (64): the run would hold no lock the box's gates and sittings take.
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
# A hold that cannot be read is read as up, never as free. The name does not match tools/ref/lease-probe.sh's LEASE_HOLDS
# (/root/bloomery-*-hold): builds and every non-GPU box command ignore it. The file has one writer, this script:
#   gpu-gate.sh --gpuhold up|beat|down <owner>   up: put it up (75 while another owner's fresh hold is up; a stale one is taken over);
#                                                beat: refresh its mtime, never create it (3: gone or another owner's);
#                                                down: remove it, only if it is <owner>'s; (owner: letters, digits, _)
#   gpu-gate.sh --gpuhold path                   print the path
# The batch calls these through tools/box.sh with BLOOMERY_BOX_READONLY=1 (no sync, no guard wait: a heartbeat stuck behind a sitting
# would go stale), a deliberate write of this one control file.
set -uo pipefail
# shellcheck source=tools/ref/lease-probe.sh
source "${BASH_SOURCE[0]%/*}/ref/lease-probe.sh"
GATE_LOCK=$LOCK_DIR/bloomery-gate.lock A6000_LOCK=$LOCK_DIR/bloomery-gate-a6000.lock LOCAL_LOCK=$LOCK_DIR/bloomery-gate-local.lock
V41_LOCK=$LOCK_DIR/bloomery-v41-load.lock V41_BOUND=1800 POLL=1 CARD_BOUND=1800
GPU_HOLD=$LOCK_DIR/bloomery-batch.gpuhold HOLD_STALE=300
V41_QDIR=$LOCK_DIR/bloomery-v41-queue.d V41_SEQ=$LOCK_DIR/bloomery-v41-queue.seq PACE=''

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

# v41_standing <ticket name>: reads the queue (batch_hold_read has just run: HOLD_STATE, HOLD_OWNER) and sets Q_N, the live tickets a fresh
# hold of another owner does not hold back, Q_POS, the named ticket's place among them (0: it is not there) and the head's item, owner
# and since. Returns 0 when the named ticket is the head.
v41_standing() {
  local i
  v41_queue_read
  Q_N=0 Q_POS=0 Q_HEAD_ITEM='' Q_HEAD_OWNER='' Q_HEAD_SINCE=''
  for ((i = 0; i < ${#Q_NAME[@]}; i++)); do
    if [ "$HOLD_STATE" = up ] && [ "${Q_OWNER[$i]}" != "$HOLD_OWNER" ]; then continue; fi
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
# on_exit: the EXIT trap of a V4.1 loader. In the self-test a paced loader (PACE) ends with one more line on its ack FIFO.
on_exit() {
  ticket_drop
  [ -z "$PACE" ] || echo exit > "$PACE.ack"
  [ -z "$PACE" ] || [ -z "${GATE_TEST_LINGER:-}" ] || sleep "$GATE_TEST_LINGER"
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
# nvidia-smi on PATH (two cards, no compute process) and a stub target/release/ok that, when asked (STUB_READY, STUB_ORDER, STUB_HOLD), says it is
# running, appends its STUB_ID to a marker file and holds until the test releases it. Concurrent runs are started with & and waited for by their $!. One line per case;
# exit 0 iff none failed.
# The cases wait on real bounds and events, so they run as SELFTEST_PARTS processes at once: each part is the cases between its
# `if gg_part K; then … fi` guards (contiguous in the file; the helpers and locals between them run in every part). `--self-test`
# starts every part and sums their counts; `--self-test --part K` runs one (K, or `all`: the parts one after another here).
SELFTEST_PARTS=3
self_test() {
  local self t n=0 bad=0 out rc rc1 p1 r1 r2 prc cap=50 part=${1:-all}
  gg_part() { [ "$part" = all ] || [ "$part" = "$1" ]; }
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  # The cases pin the box's /root paths, so a lock directory from local mode is dropped; the local cases name their own.
  unset BLOOMERY_LOCK_DIR
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
  printf '%s\n' '#!/bin/sh' '[ -z "${STUB_READY:-}" ] || echo up > "$STUB_READY"' '[ -z "${STUB_ORDER:-}" ] || echo "${STUB_ID:-?}" >> "$STUB_ORDER"' \
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
      -u BLOOMERY_TIER -u BLOOMERY_FIXTURE_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_BATCH_OWNER -u BLOOMERY_V41_CLASS -u BLOOMERY_V41_PRED_S \
      -u GATE_TEST_PACE -u STUB_ORDER -u STUB_ID \
      PATH="$t/bin:$PATH" BLOOMERY_GATE_CARD="$c" BLOOMERY_LEASE_LOCK="$l" BLOOMERY_LEASE_PROC="$t/proc" "$@" \
      bash "$self" --test-locks "${TL:-$t}" ok) > "$o" 2>&1
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
      printf '%s\n' "$out" | sed 's/^/    | /'
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
      printf '%s\n' "$out" | sed 's/^/    | /'
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
  case_ 'a6000 forced, lease cannot be tested: 70, named' 70 'cannot be tested' a6000 "$t/no-dir/lease"
  case_ 'a V4.1 load alone takes the V4.1 load lock and does not wait' 0 'ok on the 3090 \(asked 3090\), V4\.1 load lock' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=1
  case_ 'BLOOMERY_GATE_V41_LOAD other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_V41_LOAD is 1' 3090 "$t/lease" BLOOMERY_GATE_V41_LOAD=yes
  case_ 'BLOOMERY_GATE_STACKS not whole seconds: 64, named' 64 'BLOOMERY_GATE_STACKS is whole seconds' 3090 "$t/lease" BLOOMERY_GATE_STACKS=soon
  case_ 'BLOOMERY_GATE_STACKS set: the binary runs under the watch, its output through' 0 'ran on $' 3090 "$t/lease" BLOOMERY_GATE_STACKS=5
  case_ "a run closes with its own wall and exit" 0 '^gpu-gate\.sh: ok ran [0-9]+ s \(exit 0\)$' 3090 "$t/lease"
  case_ 'BLOOMERY_GATE_GDB other than 0 or 1: 64, named' 64 'BLOOMERY_GATE_GDB is 1' 3090 "$t/lease" BLOOMERY_GATE_GDB=yes
  case_ 'BLOOMERY_GATE_GDB with BLOOMERY_GATE_STACKS: 64, named' 64 'set one' 3090 "$t/lease" BLOOMERY_GATE_GDB=1 BLOOMERY_GATE_STACKS=5
  # Events, not seconds. A run in the background is held by what it waits for and released by what the test saw it do, so no case reads
  # the clock: the stub binary says `up` on STUB_READY once its run holds its locks and then blocks on STUB_HOLD until the test writes a
  # line; a waiting run's own output (read through a FIFO by relay) says it waits. The FIFOs are held open read-write by this shell
  # (descriptors 3: events, 4: the stub's release), so neither side blocks on opening one and a test that ends releases the stub by EOF.
  # A helper waiter's two bounds are a safety cap of $cap s (GATE_TEST_CAP), under gate()'s own 60 s so a waiter that hits it ends 75,
  # not 124: the test ends its life by an event, and one that hits the cap is a red by name (helper_cap). The cases that judge a bound
  # keep the runner's own 4 s and 6 s.
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
  # event <case>: blocks until one event is on descriptor 3; 60 s is gate()'s own bound on a run, so a run that never got there was killed.
  event() { read -t 60 -r _ <&3 || { n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $1: the event never came"; }; }
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
  # ack <k> <case>: ACK is loader k's next sign; 60 s is gate()'s own bound on a run, so a run that never signed was killed.
  # A loader that has signed `exit` is gone: its next sign is `gone` at once, not a minute's wait for one that cannot come. The `exit`
  # sign comes from the EXIT trap, before the process lets go of its locks (the V4.1 load lock, the card lock), so `exit` is taken as an end
  # only once the run's pid is gone: a grant to the next loader right after it would find the lock still held, and that loader would
  # poll instead of run (a sign short, then a grant it never gets, then gate()'s 60 s: rc 124).
  ack() {
    local w=0
    ACK=
    if [ "${ENDED[$1]:-0}" = 1 ]; then
      ACK=gone
      return 0
    fi
    read -t 60 -r ACK <&"$((21 + 2 * $1))" || { n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $2: loader $1 gave no sign"; ACK=none; }
    [ "$ACK" != exit ] || ENDED[$1]=1
    if [ "$ACK" = exit ] && [ -n "${LPID[$1]:-}" ]; then
      while kill -0 "${LPID[$1]}" 2> /dev/null; do
        w=$((w + 1))
        [ "$w" -lt 1200 ] || { n=$((n + 1)); bad=$((bad + 1)); echo "FAIL $2: loader $1 signed exit and was still there after 60 s"; break; }
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
  env -u BLOOMERY_HOLD_OWNER -u BLOOMERY_BOX_WAIT -u BLOOMERY_BOX_ENV -u BLOOMERY_CARD -u BLOOMERY_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_V41_MODEL \
    -u BLOOMERY_DATA -u BLOOMERY_TIER -u BLOOMERY_FIXTURE_VARIANT \
    PATH="$t/stub:$PATH" BLOOMERY_BOX=stub BLOOMERY_REMOTE='~/repo/x' BLOOMERY_BOX_READONLY=1 \
    bash "$t/boxtree/tools/box.sh" 'bash tools/gpu-gate.sh --v41-queue' > "$t/out" 2>&1 || rc=$?
  { echo "rsync calls: $(wc -l < "$t/rsync.log" | tr -d ' ')"; echo "last ssh: $(tail -1 "$t/ssh.log" | sed 's/.*&& //')"; } >> "$t/out"
  judge_all '--v41-queue runs under BLOOMERY_BOX_READONLY=1 and syncs nothing' "$rc" 0 'rsync calls: 0$' 'last ssh: bash tools/gpu-gate\.sh --v41-queue$'
  case_ 'BLOOMERY_V41_CLASS other than 1 to 4: 64, named' 64 "BLOOMERY_V41_CLASS is 1 .* got '5'" 3090 "$t/lease" BLOOMERY_V41_CLASS=5
  case_ '  … a name is no class either' 64 "BLOOMERY_V41_CLASS is 1 .* got 'two'" 3090 "$t/lease" BLOOMERY_V41_CLASS=two
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
  judge_all 'gpuhold up: takes over a stale hold, naming it' "$rc" 0 'owner other was not refreshed for 4[0-9][0-9] s: taking it over' 'after: owner=batcha since=[0-9]+ age=[0-9]+ fresh=yes$'
  rm -f "$t/batch.gpuhold"
  hrun up 'a b'
  judge_all 'gpuhold: an owner with a space: 64, named' "$rc" 64 "an owner is letters, digits and _, got 'a b'" 'after: no hold$'
  # The file's name stays apart from the sittings' holds (/root/bloomery-<owner>-hold): builds and every other box command ignore it.
  out=$(bash "$self" --gpuhold path)
  out="$out $(env -u BLOOMERY_LEASE_HOLDS bash -c '. "$1"; case /root/bloomery-03-hold in $LEASE_HOLDS) printf "sitting-hold-matches " ;; esac; case "$2" in $LEASE_HOLDS) echo MATCHES ;; *) echo apart ;; esac' _ \
    "$(dirname "$self")/ref/lease-probe.sh" "$out")"
  printf '%s\n' "$out" > "$t/out"
  judge_all "the batch hold's name does not match the sittings' LEASE_HOLDS" 0 0 '^/root/bloomery-batch\.gpuhold sitting-hold-matches apart$'
  # box.sh's local mode (BLOOMERY_BOX_CARD=local, its lock directory the test's): one lock for this machine's card(s), apart from
  # the box's two, and the caller's CUDA_VISIBLE_DEVICES (the stub prints it); a held lease is waited for as the lock is.
  fi
  local ON_LOCAL="ok on this machine's card\\(s\\), the local gate lock" LOC=(BLOOMERY_BOX_CARD=local "BLOOMERY_LOCK_DIR=$t")
  if gg_part 3; then
  rc=0
  gate "$t/out" any "$t/lease" "${LOC[@]}" CUDA_VISIBLE_DEVICES=GPU-caller || rc=$?
  judge_all 'local: any takes the local lock and leaves CUDA_VISIBLE_DEVICES as it was' "$rc" 0 "$ON_LOCAL \\(asked any\\)$" '^ran on GPU-caller$'
  case_ '  … a6000 takes it too' 0 "$ON_LOCAL \\(asked a6000\\)$" a6000 "$t/lease" "${LOC[@]}"
  case_ '  … a V4.1 load takes the V4.1 load lock with it' 0 "$ON_LOCAL \\(asked any\\), V4\\.1 load lock$" any "$t/lease" "${LOC[@]}" BLOOMERY_GATE_V41_LOAD=1
  exec 5> "$t/gate.lock" 6> "$t/gate-a6000.lock"
  { flock -x 5 && flock -x 6; } || { echo "FAIL: the test cannot hold the two card locks"; return 1; }
  case_ "  … the box's two card locks held: it runs" 0 "$ON_LOCAL \\(asked any\\)$" any "$t/lease" "${LOC[@]}"
  exec 5>&- 6>&-
  exec 6> "$t/gate-local.lock"
  flock -x 6 || { echo "FAIL: the test cannot hold the local lock"; return 1; }
  case_ '  … the local lock held: 75 at the bound' 75 'no gate lock \(local\) was free within 4 s' any "$t/lease" "${LOC[@]}"
  exec 6>&-
  exec 7> "$t/lease"
  flock -x 7 || { echo "FAIL: the test cannot hold its lease file"; return 1; }
  case_ '  … the lease held: it times the card, 75 at the bound' 75 'no gate lock \(local\) was free within 4 s' any "$t/lease" "${LOC[@]}"
  fi
  exec 7>&-
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
  trap "rm -rf '$tmp'" EXIT
  for ((k = 1; k <= SELFTEST_PARTS; k++)); do
    ( set +e; s0=$SECONDS; bash "$self" --self-test --part "$k" > "$tmp/$k.out" 2>&1; echo $? > "$tmp/$k.rc"; echo $((SECONDS - s0)) > "$tmp/$k.s" ) &
  done
  wait
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
  GATE_LOCK=$2/gate.lock A6000_LOCK=$2/gate-a6000.lock LOCAL_LOCK=$2/gate-local.lock V41_LOCK=$2/v41-load.lock GPU_HOLD=$2/batch.gpuhold V41_BOUND=6 POLL=1 CARD_BOUND=4
  V41_QDIR=$2/v41-queue.d V41_SEQ=$2/v41-queue.seq
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
elif on_box && { [ "$LOCK_DIR" != /root ] || [ "${BLOOMERY_BOX_CARD:-}" = local ]; }; then
  echo "gpu-gate.sh: $([ "$LOCK_DIR" != /root ] && echo "BLOOMERY_LOCK_DIR=$LOCK_DIR" || echo BLOOMERY_BOX_CARD=local) on the machine whose lease is $BOX_LEASE (the box): its gates and sittings lock under /root, and this run would hold none of their locks; run it through tools/box.sh without local mode. If this machine is not the box, a lock script run as root without BLOOMERY_LOCK_DIR made that file: remove it" >&2
  exit 64
fi
if [ "${1:-}" = --gpuhold ]; then
  shift
  gpuhold "$@"
  exit $?
fi
if [ "${1:-}" = --v41-queue ]; then
  v41_queue_view
  exit 0
fi
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: gpu-gate.sh <target/release binary> [args...]" >&2; exit 64; }
shift
# generate_qwen3moe counts its sources inside the binary, after this script has the card lock.
# A count other than one, and a source flag with no following word, are usage (the binary's own
# sentence) and exit 64 here, before any lock file is opened. --records-schema (record::at_main,
# exit 0) and --dump-taps (it returns before the source check) stay the binary's, lock and all.
# Exact flag names only: --prompt=foo is not a source. Repeated --arm is one source.
qwen3moe_sources() {
  local a prompt=0 tokens=0 seed=0 arm=0
  for a in "$@"; do
    case $a in
      --records-schema | --dump-taps) return 0 ;;
    esac
  done
  while [ $# -gt 0 ]; do
    a=$1
    shift
    case $a in
      --prompt) prompt=1 ;;
      --tokens) tokens=1 ;;
      --seed-depth) seed=1 ;;
      --arm) arm=1 ;;
      *) continue ;;
    esac
    if [ $# -eq 0 ]; then
      echo "generate_qwen3moe: $a needs a value" >&2
      exit 64
    fi
  done
  if [ $((prompt + tokens + seed + arm)) != 1 ]; then
    echo "generate_qwen3moe: give exactly one of --prompt, --tokens, --seed-depth, --arm" >&2
    exit 64
  fi
}
if [ "$NAME" = generate_qwen3moe ]; then
  qwen3moe_sources "$@"
fi
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
  local)
    if box_lock_dir "$LOCK_DIR"; then
      echo "gpu-gate.sh: box.sh's local mode (BLOOMERY_BOX_CARD=local) keeps its lock under BLOOMERY_LOCK_DIR, and that is $LOCK_DIR (unset: /root), the box's lock directory: run it through tools/box.sh's local mode, which sets both" >&2
      exit 64
    fi
    ASK=${BLOOMERY_GATE_CARD:-3090}
    case $ASK in
      3090 | a6000 | any) CARD=local ;;
      *) echo "gpu-gate.sh: box.sh's local mode (BLOOMERY_BOX_CARD=local) runs on what this machine shows under one lock, and BLOOMERY_GATE_CARD=$ASK names $([ "$ASK" = both ] && echo "the box's two cards" || echo 'no card'): leave it unset" >&2; exit 64 ;;
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
  local u; u=$(uuid_of A6000); [ -n "$u" ] || return 1
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"
}
# 락 파일을 못 여는 셸(박스의 root가 아니거나 박스가 아님)은 경쟁이 아니다 — 여기서 이름을 대고 끝난다.
LOCK_FILES=("9:$GATE_LOCK" "8:$A6000_LOCK")
[ "$CARD" != local ] || LOCK_FILES=("9:$LOCAL_LOCK")
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
CARD_W=0 LOAD_W=0 HOLD_W=0 why='' said=-60 hsaid=-60 csaid=-60 qsaid=-60 last=$SECONDS HAVE7=0 qnote=''
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
  if [ "$why" = hold ] && [ $((HOLD_W - hsaid)) -ge 60 ]; then
    up_s='?'
    [ -z "$HOLD_SINCE" ] || up_s=$(($(date +%s) - HOLD_SINCE))
    echo "[batch-hold] $(now) $NAME waits: a landing batch holds the GPUs ($GPU_HOLD, owner $HOLD_OWNER, up $up_s s, refreshed $HOLD_AGE s ago), holding no lock; ${HOLD_W} s so far, at most ${CARD_BOUND} s with the lock waits" >&2
    hsaid=$HOLD_W
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
echo "gpu-gate.sh: $NAME on $ON (asked $([ "$CARD" = local ] && echo "$ASK" || echo "$CARD"))$([ "$V41" = 0 ] || echo ', V4.1 load lock')$TIER_NOTE" >&2
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
