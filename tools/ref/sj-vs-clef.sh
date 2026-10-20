#!/usr/bin/env bash
# The schemajump against Clef-Flash sitting: Clef-Flash (the decide seat) and a Qwen3.6 seat (the schemajump
# jump arm), both bloomery-serve, one lease on the A6000, the arms alternating request by request.
#
#   bash tools/ref/sj-vs-clef.sh
#   bash tools/ref/sj-vs-clef.sh --self-test
#
# What it is. The user's ask (2026-10-20): schemajump — any model answering a schema the way Clef-Flash does —
# against Clef-Flash, on the time to a complete answer and on agreement. The design is
# /Users/hckim/orca/briefs/w2300/par/sjfast-files/design.md §4; the cards are docs/cards/sj-{vs-clef,jump-vs-step,
# levers,calib}-a6000.card.
#
# The windows. Two 24 GB servers do not fit the A6000 beside Clef, so the Qwen3.6 server runs in two windows of
# one lease, with Clef repeated in both as the anchor: window 1 is Clef + the Qwen3.6 jump arm (BLOOMERY_SCHEMAJUMP=on
# on the Qwen3.6 server), window 2 is Clef + the same binary with the jump arm's lever off (the step arm,
# BLOOMERY_SCHEMAJUMP=off). Both windows set the lever, so a value in the caller's environment picks no arm, and the
# Clef server runs without it. Both servers of a window stay up inside it; the Qwen3.6 server is loaded once a window
# (the load is outside every timed number).
#
# SJ_WINDOWS names the windows, in order: `<seat>:<on|off>` words, default `qwen:on qwen:off` (the two above); the
# words are split on spaces and commas, and a sitting writes commas (BLOOMERY_BOX_ENV splits on spaces).
# A `qwen` window is the Clef anchor and the Qwen3.6 server (the arms `A` and `jump` or `step`). A `ds41` window is the
# V4.1-Flash server alone (bloomery-serve-ds41, SJ_DS41_MODEL; the arm `ds41-jump` or `ds41-step`, both through
# the same `response_format`): it runs on the cards and the placement the serve binary picks (no --place), both cards
# visible, the A6000 first; the timing card pin is for the single-card seats. It starts only when neither card holds a
# compute process (the guard waits ten minutes, then ends the run at 75 naming the card and the process: another
# arm's server on a card it needs is not shared). The lever pair of a seat (`qwen:on` and `qwen:off`, `ds41:on` and
# `ds41:off`, in either order) gets the ab row below, and the table's `jump vs step` line holds their answers equal.
#
# The suites. SJ_SUITES names the suite files (words split as SJ_WINDOWS's, relative to the repository or absolute), default
# `tools/ref/clef/suite.jsonl tools/ref/clef/suite-ko.jsonl`. A `<suite>.gold.jsonl` beside a file scores every arm
# against its labels (`correct/N` per suite, the wrong picks listed); they are all read before the lease.
#
# The requests. tools/ref/clef/sjcmp.py does them: the SystemOne suites' requests converted to the Qwen seat's
# `response_format` (the schema text in the system message, the state in the user message) and posted to the
# Clef seat's `/v1/systemone` as they stand. The arms alternate request by request in a rotated order
# (ab-decode.sh's rule: arm k of round r is arm (k + r - 1) mod n), the first post of every request to every arm
# is dropped (the first-request state), and the short bucket (a prompt under 1,000 ids) runs SJ_ROUNDS rounds and
# the long bucket (the two long requests) SJ_ROUNDS_LONG. The client wall is the POST to the last byte.
#
# The bodies. A window's directory (SJ_OUT/window<n>) holds, beside rows.json, `bodies/<arm>/<suite>-<id>.json`: each
# arm's answer to each request's first counted round, as received (the Clef arm's /v1/systemone body, a generative
# arm's chat completion), the row naming it under `body`. After the last round, a generative arm whose server lists
# /v1/systemone in its /props `routes` is asked each request once more there, untimed, and the bodies go to
# `bodies/<arm>-systemone/<suite>-<id>.json` (a non-200 answer is kept with its status in `<that file>.status`); a
# server without the route is a named line, not an error. The extra requests come after every timed region.
#
# The tables. sjcmp.py's `table` prints the wall (p50, p95 per suite and per bucket) beside the server's own
# split, and the agreements (the tops against the official BF16 rows, against the Clef arm served, and the jump
# arm against the step arm). This file's own table adds the ab card's row: the jump arm's wall p50 against the
# step arm's, paired by round, with the 95 % interval of tools/ref/card.py's ruler.
#
# The lease. The run takes the machine's lease with the card (BLOOMERY_LEASE_CARD, docs/cards/sj-vs-clef-a6000.card
# for the baseline sitting and sj-jump-vs-step-a6000.card for the ab one; lease.sh checks the card and the rounds
# before the lock is opened) and the witness blocks (lease.sh's) stand around the windows. A card that cannot be
# read ends the run at 69, a busy lease at 75.
#
# The card. Timed numbers are taken on the A6000: timing-card.sh (sourced as depth-serve.sh sources it) picks the timing
# card (BLOOMERY_TIMING_GPU names another) and both servers of a window are started with that card as their only one
# (CUDA_VISIBLE_DEVICES=<its UUID> on the server's command line, which the plan prints). Before a window the runner waits
# out a compute process on the timing card (ten minutes, then 75). After a window it reads the card again (it must answer
# nvidia-smi by its UUID) and each server's log: a plan or lane record that names a card other than the timing card ends
# the run (the Clef seat's log names none, which a line says). timing_cards_arm is called too, but its Xid and lost-card
# reads belong to the two-card and local modes (timing-card.sh), which this runner does not opt into: in the one-card mode
# it returns at once, and the two reads above are the runner's own. A server binary older than its sources is refused (3)
# before the lease.
#
# The bounds. Every process the run starts under the lease has a bound (tools/gate-bound.sh): a server is a
# bounded_bg command under a handle of its own (lcpp-warm.sh's lcpp_srv_start, SJ_SERVER_BOUND covers its load and
# its window), the request run and the tables are lease_bounded (SJ_RUN_BOUND for the run). A TERM or INT to the
# runner, or a refusal, stops both servers and the run, and ends the lease.
#
# The knobs. None is a BLOOMERY_ name, which would have to be a row of the lever registry though no binary reads
# it; a BLOOMERY_SERVE_ variable set in the environment is refused by name.
#   SJ_CLEF_MODEL        the Clef model file (the head is in the file); a `qwen` window needs it
#   SJ_CLEF_HEAD         the head file, when the model carries none
#   SJ_QWEN_MODEL        the Qwen3.6 model file; a `qwen` window needs it
#   SJ_DS41_MODEL        the V4.1-Flash model file (the first shard); a `ds41` window needs it
#   SJ_DS41_BIN          the V4.1 server binary, default target/release/bloomery-serve-ds41
#   SJ_DS41_PORT         the V4.1 server's port, default 18183
#   SJ_WINDOWS           the windows, default `qwen:on qwen:off` (above)
#   SJ_SUITES            the suite files, default the English and Korean ones (above)
#   SJ_CTX               the servers' --ctx-size, default 16384
#   SJ_CLEF_PORT         the Clef server's port, default 18181
#   SJ_QWEN_PORT         the Qwen3.6 server's port, default 18182
#   SJ_ROUNDS            the short bucket's rounds, default 20 (BLOOMERY_AB_ROUNDS overrides it)
#   SJ_ROUNDS_LONG       the long bucket's rounds, default 8
#   SJ_SERVER_BOUND      a server's life in seconds, its load and its window, default BLOOMERY_ARM_BOUND (900)
#   SJ_RUN_BOUND         the request run's seconds, default 600; under SJ_SERVER_BOUND
#   SJ_OUT               where the rows go, default a fresh directory under TMPDIR
#   SJ_LOGDIR            where the servers' logs go, default SJ_OUT
#   SJ_SERVE_BIN         the server binary, default target/release/bloomery-serve
#
# Refused by name: no model file (64), a model file that is not there (2), a port that is not a port (64), a round
# count or a bound that is not a whole number (64), a run bound that does not fit inside the server's (64), a window
# word that is not one (64), a suite or gold file that does not read (64), a missing server binary (2), a stale one
# (3), a timing card timing-card.sh refused (64) or one that does not answer (69), a card a ds41 window needs held
# by another process (75). Under
# BLOOMERY_DRY=1 the plan is printed and nothing is started, no lease is taken; a missing or stale server binary is a
# note there, since a dry run builds nothing.
set -uo pipefail
__d=${BASH_SOURCE[0]%/*}
[ "$__d" != "${BASH_SOURCE[0]}" ] || __d=.
ROOT=$(cd "$__d/../.." && pwd)
HERE=$ROOT/tools/ref
unset __d
SJ_PY=$HERE/clef/sjcmp.py
CARD_PY=$HERE/card.py
SERVE_BIN=target/release/bloomery-serve
CLEF_MODEL=''
CLEF_HEAD=''
QWEN_MODEL=''
DS41_MODEL=''
DS41_BIN=target/release/bloomery-serve-ds41
CTX=16384
CLEF_PORT=18181
QWEN_PORT=18182
DS41_PORT=18183
WINDOWS=''
SUITE_FILES=''
ROUNDS=20
ROUNDS_LONG=8
SERVER_BOUND=''
RUN_BOUND=''
TOOL_BOUND=120
OUT=''
LOGDIR=''
# The timing card's name (sj_card_name) and the reason the last card or log check refused.
TIMING_NAME=''
SJ_WHY=''

# sj_read_knobs: the knobs, read when the run starts (the self-test sets them in its own environment).
sj_read_knobs() {
  SERVE_BIN=${SJ_SERVE_BIN:-target/release/bloomery-serve}
  CLEF_MODEL=${SJ_CLEF_MODEL:-}
  CLEF_HEAD=${SJ_CLEF_HEAD:-}
  QWEN_MODEL=${SJ_QWEN_MODEL:-}
  DS41_MODEL=${SJ_DS41_MODEL:-}
  DS41_BIN=${SJ_DS41_BIN:-target/release/bloomery-serve-ds41}
  CTX=${SJ_CTX:-16384}
  CLEF_PORT=${SJ_CLEF_PORT:-18181}
  QWEN_PORT=${SJ_QWEN_PORT:-18182}
  DS41_PORT=${SJ_DS41_PORT:-18183}
  # Words split on spaces or commas: BLOOMERY_BOX_ENV is split on spaces, so a sitting writes commas.
  WINDOWS=${SJ_WINDOWS:-qwen:on qwen:off}
  WINDOWS=${WINDOWS//,/ }
  SUITE_FILES=${SJ_SUITES:-tools/ref/clef/suite.jsonl tools/ref/clef/suite-ko.jsonl}
  SUITE_FILES=${SUITE_FILES//,/ }
  ROUNDS=${SJ_ROUNDS:-20}
  ROUNDS=${BLOOMERY_AB_ROUNDS:-$ROUNDS}
  ROUNDS_LONG=${SJ_ROUNDS_LONG:-8}
  SERVER_BOUND=${SJ_SERVER_BOUND:-$LEASE_ARM_BOUND}
  RUN_BOUND=${SJ_RUN_BOUND:-600}
  OUT=${SJ_OUT:-}
  LOGDIR=${SJ_LOGDIR:-}
}

MODE=run
case ${1:-} in
  --self-test) MODE=selftest; shift ;;
esac

# The lease, the witness blocks and, through it, the bounds (tools/gate-bound.sh); the timing card and the stale-binary
# rule (the self-test's own process runs no sitting and sources the lease alone: its cases source this file again); the
# servers' start (lcpp-warm.sh).
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"
if [ "$MODE" != selftest ]; then
  # shellcheck source=tools/ref/timing-card.sh
  source "$HERE/timing-card.sh"
fi
# shellcheck source=tools/ref/lcpp-warm.sh
source "$HERE/lcpp-warm.sh"

sj_refuse() {
  echo "sj-vs-clef.sh: $2" >&2
  exit "$1"
}

sj_posint() {
  case $2 in '' | *[!0-9]*) sj_refuse 64 "$1 is a whole number, got '$2'" ;; esac
  [ "$2" -ge "$3" ] || sj_refuse 64 "$1 is at least $3, got $2"
}

# The servers of a window run as bounded_bg commands, each under a handle of its own (clef, qwen; tools/gate-bound.sh):
# their pids are the owner's, a TERM or INT to the runner stops both with their groups, and sj_stop_servers (a window's
# end, and the EXIT trap) stops what is still up.
sj_stop_servers() {
  bounded_bg_stop_all || sj_refuse 70 "a server outlived its cleanup budget (tools/gate-bound.sh printed which)"
}

# The EXIT trap: both servers stopped on every way out, a refusal included, and the lease runner's capture directory
# removed before an exit of this function's own (lease_out_clean; lease_out_init appends it to the trap and would
# not be reached after one). The trap's text names this function only.
sj_cleanup() {
  local rc=$?
  trap - EXIT
  bounded_bg_stop_all || rc=70
  lease_out_clean
  exit "$rc"
}

# sj_start <name> <model> <port> <log> [head] [env words...]: the server under the handle <name>, its log and the
# line that says it listens (lcpp_srv_start). The head file belongs to the Clef seat alone. The env words go to `env`
# after -u BLOOMERY_REF_MODEL: the model is named once, by -m. The timing card is the server's only card, whatever
# CUDA_VISIBLE_DEVICES the caller holds.
sj_start() {
  local name=$1 model=$2 port=$3 log=$4 head=$5
  shift 5
  SRV_CMD=("$SERVE_BIN" -m "$model")
  [ -z "$head" ] || SRV_CMD+=(--head "$head")
  SRV_CMD+=(--host 127.0.0.1 --port "$port" --ctx-size "$CTX")
  SRV_HANDLE=$name
  if ! lcpp_srv_start "$SERVER_BOUND" "$log" -u BLOOMERY_REF_MODEL "$@" "CUDA_VISIBLE_DEVICES=$TIMING_GPU"; then
    echo "--- the $name server's log ($log), last 20 lines" >&2
    tail -n 20 "$log" >&2
    sj_refuse 1 "the $name server: $SRV_WHY"
  fi
  sj_started "$name" "$port" "$log"
}

# sj_started <name> <port> <log>: the started server listens on the port it was given and its log proves the context.
sj_started() {
  local name=$1 port=$2 log=$3 line ctx
  [ "$SRV_PORT" = "$port" ] || sj_refuse 1 "the $name server listens on port $SRV_PORT, not the --port $port it was given"
  line=$(grep -m1 'listening on http://' "$log")
  ctx=$(sed -n 's/.* ctx=\([0-9][0-9]*\) .*/\1/p' <<< "$line")
  # The decide seat's listening line names no context; its `ctx rule=` record, printed before it, does.
  [ -n "$ctx" ] || ctx=$(sed -n 's/^ctx rule=[a-z]* ctx=\([0-9][0-9]*\) .*/\1/p' "$log" | head -n 1)
  [ -n "$ctx" ] || sj_refuse 1 "the $name server's log names no ctx= (listening line or ctx rule record), so the context the arm ran at is not proven"
  [ "$ctx" = "$CTX" ] || sj_refuse 1 "the $name server's line names ctx=$ctx, not the --ctx-size $CTX the arm asked"
}

# sj_ds41_cards: the cards the V4.1 server is given, the timing card first (the plan names which it uses).
sj_ds41_cards() {
  echo "$TIMING_GPU,$OTHER_GPU"
}

# sj_start_ds41 <log> <lever>: the V4.1 server under the handle ds41. The model rides BLOOMERY_REF_MODEL (the seat reads
# no -m), there is no --place (the binary picks its placement and cards), and both cards are visible, the timing card
# first: the single-card pin of the other seats is not this one's.
sj_start_ds41() {
  local log=$1 lever=$2
  SRV_CMD=("$DS41_BIN" --host 127.0.0.1 --port "$DS41_PORT" --ctx-size "$CTX")
  SRV_HANDLE=ds41
  if ! lcpp_srv_start "$SERVER_BOUND" "$log" "BLOOMERY_REF_MODEL=$DS41_MODEL" "BLOOMERY_SCHEMAJUMP=$lever" \
    "CUDA_VISIBLE_DEVICES=$(sj_ds41_cards)"; then
    echo "--- the ds41 server's log ($log), last 20 lines" >&2
    tail -n 20 "$log" >&2
    sj_refuse 1 "the ds41 server: $SRV_WHY"
  fi
  sj_started ds41 "$DS41_PORT" "$log"
}

# sj_card_name: the timing card's name as nvidia-smi prints it, asked by its UUID, in TIMING_NAME; 1 with SJ_WHY when the
# card does not answer (a card off the bus fails or hangs the query, which is bounded).
sj_card_name() {
  local rc=0
  TIMING_NAME=''
  lease_capture 30 nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" || rc=$?
  if [ "$rc" != 0 ] || [ -z "$LEASE_OUT" ]; then
    SJ_WHY="the timing card ($TIMING_GPU) does not answer nvidia-smi (rc $rc)"
    return 1
  fi
  TIMING_NAME=${LEASE_OUT%%$'\n'*}
}

# sj_log_cards <log>: the cards a server's log names, one a line: the `card=` word of its plan and lane records and the
# names in the plan's `devices=[role:name:cudaN:bytes,...]` list.
sj_log_cards() {
  awk '
    $1 == "plan" || $1 == "lane" {
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^card=/) print substr($i, 6)
        if ($i ~ /^devices=\[/) {
          list = $i
          sub(/^devices=\[/, "", list)
          sub(/\].*$/, "", list)
          n = split(list, dev, ",")
          for (j = 1; j <= n; j++) { split(dev[j], f, ":"); print f[2] }
        }
      }
    }' "$1" | sort -u
}

# sj_guard_timing <tag>: a compute process on the timing card as a window starts is another round's functional run (a
# functional run is refused while the card is busy, not while the lease is held, so one can start between two windows).
# The servers would load onto that card, so wait for it, SJ_BUSY_POLLS polls SJ_BUSY_STEP seconds apart (ten minutes),
# and refuse after that, rc 75: contention, not a result. A card that does not answer is refused at once, 69.
SJ_BUSY_POLLS=60 SJ_BUSY_STEP=10
# sj_busy_apps <tag> <card uuid>...: the compute processes on the cards, `the timing card (UUID): [apps]` words joined
# by `; `, in SJ_BUSY_APPS (empty when every card is free). A card that does not answer ends the run at 69.
sj_busy_apps() {
  local tag=$1 card rc label
  shift
  SJ_BUSY_APPS=''
  for card in "$@"; do
    if [ "$card" = "$TIMING_GPU" ]; then label='the timing card'; else label='the other card'; fi
    rc=0
    lease_capture 30 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$card" || rc=$?
    [ "$rc" = 0 ] || sj_refuse 69 "$tag: $label ($card) does not answer nvidia-smi (rc $rc)"
    [ -n "$LEASE_OUT" ] || continue
    SJ_BUSY_APPS+="${SJ_BUSY_APPS:+; }compute apps on $label ($card): [$(echo "$LEASE_OUT" | tr '\n' ';')]"
  done
}

# sj_guard_cards <tag> <need> <card uuid>...: wait out the compute processes on the cards (SJ_BUSY_POLLS polls
# SJ_BUSY_STEP seconds apart), and end the run at 75 when they stay. <need> is the sentence the final refusal adds.
sj_guard_cards() {
  local tag=$1 need=$2 i
  shift 2
  for ((i = 0; i < SJ_BUSY_POLLS; i++)); do
    sj_busy_apps "$tag" "$@"
    if [ -z "$SJ_BUSY_APPS" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * SJ_BUSY_STEP)) s" >&2
      return 0
    fi
    if [ "$i" = 0 ]; then
      echo "[timing-busy] $(now) $tag: $SJ_BUSY_APPS; waiting up to $((SJ_BUSY_POLLS * SJ_BUSY_STEP)) s" >&2
      witness wait-timing >&2
    fi
    sleep "$SJ_BUSY_STEP"
  done
  echo "[timing-busy] $(now) still busy after $((SJ_BUSY_POLLS * SJ_BUSY_STEP)) s: $SJ_BUSY_APPS${need:+ ($need)}" >&2
  witness abort-timing >&2
  exit 75
}

# sj_guard_timing <tag>: the timing card, for the single-card seats.
sj_guard_timing() {
  sj_guard_cards "$1" '' "$TIMING_GPU"
}

# sj_guard_ds41 <tag>: both cards, for the V4.1 server, which takes the cards its placement picks; a process of another
# arm's server on either one ends the run by name.
sj_guard_ds41() {
  [ -n "$OTHER_GPU" ] || sj_refuse 69 "$1: the other card has no UUID (tools/ref/cards.sh${CARDS_ERROR:+: $CARDS_ERROR}), so the V4.1 server's cards are not proven free"
  sj_guard_cards "$1" 'the V4.1 server needs both cards and starts on neither while another process holds one' "$TIMING_GPU" "$OTHER_GPU"
}

# sj_card_check <n>: after window <n>'s servers stopped, the timing card still answers and neither server's log names
# another card. 1 with SJ_WHY otherwise.
sj_card_check() {
  local n=$1 seat=${2:-qwen} s log names x place
  sj_card_name || { SJ_WHY="window $n: $SJ_WHY"; return 1; }
  if [ "$seat" = ds41 ]; then
    # The V4.1 server takes the cards its placement picks: the log's names are recorded, none is refused.
    names=$(sj_log_cards "$LOGDIR/ds41-w$n.log" | tr '\n' ' ' | sed 's/ $//')
    place=$(sed -n 's/^plan place=\([^ ]*\).*/\1/p' "$LOGDIR/ds41-w$n.log" | head -n 1)
    echo "[card] window $n ds41: the log names ${names:-no card}${place:+ (plan place=$place)}; the server was given CUDA_VISIBLE_DEVICES=$(sj_ds41_cards), the timing card ($TIMING_NAME) first"
    return 0
  fi
  for s in clef qwen; do
    log=$LOGDIR/$s-w$n.log
    timing_cards_arm "$(cat "$log")" || { SJ_WHY="window $n: the $s server: $TWOCARD_WHY"; return 1; }
    names=$(sj_log_cards "$log")
    if [ -z "$names" ]; then
      echo "[card] window $n $s: the log names no card; CUDA_VISIBLE_DEVICES=$TIMING_GPU pins it"
      continue
    fi
    for x in $names; do
      case $TIMING_NAME in
        *"$x"*) ;;
        *) SJ_WHY="window $n: the $s server's log names the card $x, and the timing card is $TIMING_NAME"; return 1 ;;
      esac
    done
    echo "[card] window $n $s: the log names $(tr '\n' ' ' <<< "$names" | sed 's/ $//'), the timing card ($TIMING_NAME)"
  done
}

# sj_window <n> <seat> <lever>: one window's servers, the rounds, the rows. A `qwen` window starts the Clef server
# (without the lever) and the Qwen3.6 server (with it on or off), the arms `A` and `jump` or `step`; a `ds41` window
# starts the V4.1 server alone (the arm `ds41-jump` or `ds41-step`).
sj_window() {
  local n=$1 seat=$2 lever=$3 arm port why='' f
  local dir="$OUT/window$n"
  local args=()
  case $seat in
    qwen) arm=jump port=$QWEN_PORT ;;
    *) arm=ds41-jump port=$DS41_PORT ;;
  esac
  [ "$lever" = on ] || arm=${arm%jump}step
  mkdir -p "$dir"
  CPU_BUSY_TAG='' OTHER_BUSY_TAG=''
  if [ "$seat" = qwen ]; then
    guard_other
    sj_guard_timing "pre window $n"
    guard_cpu "pre window $n"
    sj_start clef "$CLEF_MODEL" "$CLEF_PORT" "$LOGDIR/clef-w$n.log" "$CLEF_HEAD" -u BLOOMERY_SCHEMAJUMP
    sj_start qwen "$QWEN_MODEL" "$QWEN_PORT" "$LOGDIR/qwen-w$n.log" "" "BLOOMERY_SCHEMAJUMP=$lever"
    args+=(--clef "A=http://127.0.0.1:$CLEF_PORT")
  else
    sj_guard_ds41 "pre window $n"
    guard_cpu "pre window $n"
    sj_start_ds41 "$LOGDIR/ds41-w$n.log" "$lever"
  fi
  args+=(--qwen "$arm=http://127.0.0.1:$port")
  for f in $SUITE_FILES; do args+=(--suite "$f"); done
  if ! lease_bounded "$RUN_BOUND" python3 "$SJ_PY" run "${args[@]}" --rounds "$ROUNDS" --rounds-long "$ROUNDS_LONG" --out "$dir"; then
    # A card that fell off the bus is the likelier reason a request run fails: name it beside the runner's.
    sj_stop_servers
    sj_card_check "$n" "$seat" || why="; $SJ_WHY"
    sj_refuse 1 "the requests of window $n failed (the runner's reason is above)$why"
  fi
  sj_stop_servers
  sj_card_check "$n" "$seat" || sj_refuse 1 "$SJ_WHY"
  guard_cpu "post window $n"
  if [ "$seat" = qwen ]; then
    echo "[window $n] the Clef arm and the $arm arm: $ROUNDS short rounds, $ROUNDS_LONG long rounds, rows in $dir" \
      "${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}"
  else
    echo "[window $n] the $arm arm: $ROUNDS short rounds, $ROUNDS_LONG long rounds, rows in $dir" \
      "${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}"
  fi
}

# The ab row: one arm's wall p50 against another's, paired by round, with the ruler's interval (the measured effect's
# own, and the card's ruler beside it: card.py's half_width at the card's rounds and sd).
# sj_ab <rows A> <arm A|-> <label A> <rows B> <arm B|-> <label B> <buckets> <card file|->: `-` for an arm finds it by the
# shell's names (B or jump, C or step); the buckets are the comma-separated ones to print (a bucket no round holds both
# arms in is passed over; none at all ends the run); `-` for the card prints no ruler of the card's.
sj_ab() {
  lease_bounded "$TOOL_BOUND" python3 -I - "$CARD_PY" "$@" << 'PY'
import importlib.util
import json
import statistics
import sys

spec = importlib.util.spec_from_file_location('card', sys.argv[1])
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)
file_a, arm_a, label_a, file_b, arm_b, label_b, buckets, card_file = sys.argv[2:10]


def arms(path):
    body = json.loads(open(path).read())
    out = {}
    for r in body['rows']:
        if r.get('dropped'):
            continue
        out.setdefault(r['arm'], []).append(r)
    return out


def p50(rows, rnd, bucket):
    walls = [r['wall_ms'] for r in rows if r['round'] == rnd and r['bucket'] == bucket]
    return statistics.median(walls) if walls else None


a, b = arms(file_a), arms(file_b)
if arm_a == '-':
    arm_a = next((k for k in a if k.lower() in ('b', 'jump')), None)
if arm_b == '-':
    arm_b = next((k for k in b if k.lower() in ('c', 'step')), None)
if arm_a not in a or arm_b not in b:
    print(f'sj-vs-clef.sh: the two windows do not hold a {label_a} arm and a {label_b} arm: '
          f'{sorted(a)} and {sorted(b)}')
    sys.exit(1)
t975 = card.t_table()
shown = 0
for bucket in buckets.split(','):
    rounds = sorted({r['round'] for r in a[arm_a] if r['bucket'] == bucket} &
                    {r['round'] for r in b[arm_b] if r['bucket'] == bucket})
    diffs = []
    for rnd in rounds:
        x, y = p50(a[arm_a], rnd, bucket), p50(b[arm_b], rnd, bucket)
        if x is None or y is None:
            continue
        diffs.append(100.0 * (x - y) / y)
    if not diffs:
        continue
    shown += 1
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    h, t, df = card.paired_half_width(t975, sd, len(diffs))
    print(f'=== the {label_a} arm against the {label_b} arm (the {bucket} bucket\'s wall p50, paired by round) ===')
    print(f'{label_a} {arm_a} p50 {statistics.median([p50(a[arm_a], r, bucket) for r in rounds if p50(a[arm_a], r, bucket) is not None]):.1f} ms | '
          f'{label_b} {arm_b} p50 {statistics.median([p50(b[arm_b], r, bucket) for r in rounds if p50(b[arm_b], r, bucket) is not None]):.1f} ms | '
          f'rounds {len(diffs)}')
    print(f'diff ({label_a} - {label_b}) / {label_b} {mean:+.2f} % (sd {sd:.2f} %) +- {h:.2f} % '
          f'(t {t:.3f}, df {df}; card.py paired_half_width) [{"clears 0" if abs(mean) > h else "holds 0"}]')
    if card_file == '-':
        continue
    if card_file:
        try:
            c = card.validate(card.parse(open(card_file).read()))
        except OSError as e:
            print(f"the card at {card_file} is not readable ({e}): the card's ruler is not shown")
        except card.Refused as e:
            print(f"the card at {card_file} is refused ({e}): the card's ruler is not shown")
        else:
            if 'sd' in c and 'rounds' in c:
                csd, csd_source = c['sd']
                h2, t2, df2 = card.half_width(t975, csd, c['rounds'])
                print(f"the card's ruler: h = {h2:.3f} % at {c['rounds']} rounds (t {t2:.3f}, df {df2}; "
                      f"sd {csd:g} % — {csd_source}) — the effect {'clears' if abs(mean) > h2 else 'holds'} it")
            else:
                print(f"the card at {card_file} is a {c['kind']} card: it has no ruler")
    else:
        print("no card named: the card's ruler is not shown")
if not shown:
    print('sj-vs-clef.sh: no round holds both arms')
    sys.exit(1)
PY
}

# sj_ruler [rows A rows B]: the jump arm against the step arm of the Qwen seat, the short bucket, the lease's card
# beside it. The two windows are window 1 and window 2 unless named.
sj_ruler() {
  sj_ab "${1:-$OUT/window1/rows.json}" - jump "${2:-$OUT/window2/rows.json}" - step short "${BLOOMERY_LEASE_CARD:-}"
}

# sj_ds41_ruler <rows jump> <rows step>: the V4.1 seat's jump arm against its step arm, the short and the long bucket,
# each paired by round. The lease's card is the Qwen pair's, so no card ruler stands beside it.
sj_ds41_ruler() {
  sj_ab "$1" ds41-jump 'ds41 jump' "$2" ds41-step 'ds41 step' short,long -
}

# sj_win_parse <word>: a window word `<seat>:<on|off>` into W_SEAT and W_LEVER; a word that is not one ends the run
# at 64.
sj_win_parse() {
  local word=$1 rest=$1
  case $rest in *:*) ;; *) sj_refuse 64 "SJ_WINDOWS word '$word' is <seat>:<on|off>" ;; esac
  W_SEAT=${rest%%:*} W_LEVER=${rest#*:}
  case $W_SEAT in qwen | ds41) ;; *) sj_refuse 64 "SJ_WINDOWS word '$word' names the seat '$W_SEAT': qwen or ds41" ;; esac
  case $W_LEVER in on | off) ;; *) sj_refuse 64 "SJ_WINDOWS word '$word' names the lever '$W_LEVER': on or off" ;; esac
}

# sj_windows_read: SJ_WINDOWS into the arrays WSEAT and WLEVER (one entry a window), the seats in NEED_QWEN and
# NEED_DS41. A word twice, and no word, end the run at 64: the same arm in two windows would be one arm in the tables.
sj_windows_read() {
  local word seen=' '
  WSEAT=() WLEVER=()
  NEED_QWEN='' NEED_DS41=''
  for word in $WINDOWS; do
    sj_win_parse "$word"
    case $seen in *" ${W_SEAT}:${W_LEVER} "*) sj_refuse 64 "SJ_WINDOWS names ${W_SEAT}:${W_LEVER} twice: its arm would be two arms of one name" ;; esac
    seen+="${W_SEAT}:${W_LEVER} "
    WSEAT+=("$W_SEAT") WLEVER+=("$W_LEVER")
    if [ "$W_SEAT" = qwen ]; then NEED_QWEN=1; else NEED_DS41=1; fi
  done
  [ "${#WSEAT[@]}" -gt 0 ] || sj_refuse 64 "SJ_WINDOWS names no window"
}

# sj_print_plan: the plan, for a dry run and the head of a real one: the servers' command lines of every window and
# the bounds.
sj_print_plan() {
  local n lever
  echo "[plan] clef=$CLEF_MODEL${CLEF_HEAD:+ head=$CLEF_HEAD} qwen=$QWEN_MODEL ds41=${DS41_MODEL:-none} ctx=$CTX ports=$CLEF_PORT/$QWEN_PORT/$DS41_PORT rounds=$ROUNDS/$ROUNDS_LONG out=$OUT"
  echo "[plan] windows: $WINDOWS suites: $SUITE_FILES"
  echo "[plan] bounds: server ${SERVER_BOUND}s (load and window), run ${RUN_BOUND}s, tables ${TOOL_BOUND}s"
  echo "[plan] card: timing_gpu=$TIMING_GPU (${TIMING_NAME:-no name: $SJ_WHY}) other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for ((n = 1; n <= ${#WSEAT[@]}; n++)); do
    lever=${WLEVER[n - 1]}
    if [ "${WSEAT[n - 1]}" = qwen ]; then
      echo "[plan] window $n clef: timeout $SERVER_BOUND env -u BLOOMERY_REF_MODEL -u BLOOMERY_SCHEMAJUMP CUDA_VISIBLE_DEVICES=$TIMING_GPU $SERVE_BIN -m $CLEF_MODEL${CLEF_HEAD:+ --head $CLEF_HEAD} --host 127.0.0.1 --port $CLEF_PORT --ctx-size $CTX"
      echo "[plan] window $n qwen: timeout $SERVER_BOUND env -u BLOOMERY_REF_MODEL BLOOMERY_SCHEMAJUMP=$lever CUDA_VISIBLE_DEVICES=$TIMING_GPU $SERVE_BIN -m $QWEN_MODEL --host 127.0.0.1 --port $QWEN_PORT --ctx-size $CTX"
    else
      echo "[plan] window $n ds41: timeout $SERVER_BOUND env BLOOMERY_REF_MODEL=$DS41_MODEL BLOOMERY_SCHEMAJUMP=$lever CUDA_VISIBLE_DEVICES=$(sj_ds41_cards) $DS41_BIN --host 127.0.0.1 --port $DS41_PORT --ctx-size $CTX"
    fi
    echo "[plan] window $n arms: ${WSEAT[n - 1]}:$lever"
  done
}

sj_main() {
  local v rc n bin seat on_n off_n
  local suite_args=() tables=()
  sj_read_knobs
  for v in $(compgen -v BLOOMERY_SERVE_); do
    sj_refuse 64 "$v is set, and nothing reads it: this runner's knobs are SJ_CLEF_MODEL, SJ_CLEF_HEAD, SJ_QWEN_MODEL, SJ_DS41_MODEL, SJ_DS41_BIN, SJ_DS41_PORT, SJ_WINDOWS, SJ_SUITES, SJ_CTX, SJ_CLEF_PORT, SJ_QWEN_PORT, SJ_ROUNDS, SJ_ROUNDS_LONG, SJ_SERVER_BOUND, SJ_RUN_BOUND, SJ_OUT, SJ_LOGDIR and SJ_SERVE_BIN"
  done
  sj_windows_read
  sj_posint ROUNDS "$ROUNDS" 1
  sj_posint SJ_ROUNDS_LONG "$ROUNDS_LONG" 1
  sj_posint SJ_CTX "$CTX" 1
  sj_posint SJ_CLEF_PORT "$CLEF_PORT" 1
  sj_posint SJ_QWEN_PORT "$QWEN_PORT" 1
  sj_posint SJ_DS41_PORT "$DS41_PORT" 1
  sj_posint SJ_SERVER_BOUND "$SERVER_BOUND" 1
  sj_posint SJ_RUN_BOUND "$RUN_BOUND" 1
  [ "$RUN_BOUND" -lt "$SERVER_BOUND" ] ||
    sj_refuse 64 "SJ_RUN_BOUND $RUN_BOUND s is inside SJ_SERVER_BOUND $SERVER_BOUND s: a server lives its load and the run"
  [ "$CLEF_PORT" -le 65535 ] || sj_refuse 64 "SJ_CLEF_PORT is a port, got $CLEF_PORT"
  [ "$QWEN_PORT" -le 65535 ] || sj_refuse 64 "SJ_QWEN_PORT is a port, got $QWEN_PORT"
  [ "$DS41_PORT" -le 65535 ] || sj_refuse 64 "SJ_DS41_PORT is a port, got $DS41_PORT"
  [ "$CLEF_PORT" != "$QWEN_PORT" ] || sj_refuse 64 "the two servers want the same port $CLEF_PORT"
  if [ -n "$NEED_QWEN" ]; then
    [ -n "$CLEF_MODEL" ] || sj_refuse 64 "SJ_CLEF_MODEL names the Clef model file"
    [ -n "$QWEN_MODEL" ] || sj_refuse 64 "SJ_QWEN_MODEL names the Qwen3.6 model file"
    [ -e "$CLEF_MODEL" ] || sj_refuse 2 "no Clef model file at $CLEF_MODEL"
    [ -e "$QWEN_MODEL" ] || sj_refuse 2 "no Qwen3.6 model file at $QWEN_MODEL"
  fi
  if [ -n "$NEED_DS41" ]; then
    [ -n "$DS41_MODEL" ] || sj_refuse 64 "SJ_DS41_MODEL names the V4.1 model file (the first shard)"
    [ -e "$DS41_MODEL" ] || sj_refuse 2 "no V4.1 model file at $DS41_MODEL"
  fi
  # A card pick timing-card.sh refused (it printed why as this file was sourced) ends the run here, dry or not.
  [ -z "$TIMING_CARDS_WHY" ] || sj_refuse 64 "no timing card: $TIMING_CARDS_WHY"
  # A dry run builds nothing, so a tree with no binary yet, or one older than its sources, still prints its plan; a real
  # run is refused. assert_fresh_binary walks the tree it stands in: the repository root, where the binaries are
  # relative to. Each seat the windows name has its own binary.
  BIN_NOTE=''
  for bin in ${NEED_QWEN:+"$SERVE_BIN"} ${NEED_DS41:+"$DS41_BIN"}; do
    if [ ! -x "$bin" ]; then
      [ -n "${BLOOMERY_DRY:-}" ] || sj_refuse 2 "no server binary at $bin (the recipe builds it)"
      BIN_NOTE+="${BIN_NOTE:+; }no server binary at $bin: a real run is refused (rc 2); the recipe builds it"
    else
      rc=0
      assert_fresh_binary "$bin" || rc=$?
      if [ "$rc" != 0 ]; then
        [ -n "${BLOOMERY_DRY:-}" ] || exit "$rc"
        BIN_NOTE+="${BIN_NOTE:+; }$bin would be refused (rc $rc)"
      fi
    fi
  done
  [ -r "$SJ_PY" ] || sj_refuse 2 "no runner at $SJ_PY"
  # The suites and their gold are read before anything is planned on a card: a file that does not load, a gold line
  # that names what the suite lacks, ends the run here, dry or not.
  for v in $SUITE_FILES; do suite_args+=(--suite "$v"); done
  [ "${#suite_args[@]}" -gt 0 ] || sj_refuse 64 "SJ_SUITES names no suite file"
  rc=0
  lease_capture "$TOOL_BOUND" python3 "$SJ_PY" check-suites "${suite_args[@]}" || rc=$?
  [ "$rc" = 0 ] || sj_refuse 64 "the suites do not load (rc $rc): $LEASE_OUT"
  SUITE_NOTE=$(tr '\n' ';' <<< "$LEASE_OUT")
  if [ -z "$OUT" ]; then
    tmp=${TMPDIR:-/tmp}
    OUT=$(mktemp -d "${tmp%/}/sj-vs-clef.XXXXXX") || sj_refuse 2 "no temporary directory"
  fi
  mkdir -p "$OUT" || sj_refuse 2 "no output directory at $OUT"
  if [ -z "$LOGDIR" ]; then
    LOGDIR=$OUT
  fi
  mkdir -p "$LOGDIR" || sj_refuse 2 "no log directory at $LOGDIR"
  # The card answers by its UUID before anything is planned on it; a dry run prints why it did not and goes on.
  if ! sj_card_name; then
    [ -n "${BLOOMERY_DRY:-}" ] || sj_refuse 69 "refused before the lease: $SJ_WHY"
  fi
  sj_print_plan
  echo "[plan] suites read: $SUITE_NOTE"
  if [ -n "${BLOOMERY_DRY:-}" ]; then
    [ -z "$BIN_NOTE" ] || echo "[dry] $BIN_NOTE"
    echo "[dry] nothing built, no lease taken, no server started"
    return 0
  fi
  trap sj_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  WITNESS=(head-load indent card busiest model mem pgmajfault)
  timing_cards_before_lease
  lease_take
  timing_cards_start
  echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES rounds=$ROUNDS/$ROUNDS_LONG logs=$LOGDIR"
  witness pre
  guard_other
  for ((n = 1; n <= ${#WSEAT[@]}; n++)); do
    sj_window "$n" "${WSEAT[n - 1]}" "${WLEVER[n - 1]}"
  done
  witness post
  for ((n = 1; n <= ${#WSEAT[@]}; n++)); do tables+=("$OUT/window$n/rows.json"); done
  lease_bounded "$TOOL_BOUND" python3 "$SJ_PY" table "${tables[@]}" ||
    sj_refuse 1 "the tables failed"
  # The ab rows: each seat's lever pair, in whichever windows they stand.
  for seat in qwen ds41; do
    on_n='' off_n=''
    for ((n = 1; n <= ${#WSEAT[@]}; n++)); do
      [ "${WSEAT[n - 1]}" = "$seat" ] || continue
      if [ "${WLEVER[n - 1]}" = on ]; then on_n=$n; else off_n=$n; fi
    done
    [ -n "$on_n" ] && [ -n "$off_n" ] || continue
    if [ "$seat" = qwen ]; then
      sj_ruler "$OUT/window$on_n/rows.json" "$OUT/window$off_n/rows.json" || sj_refuse 1 "the ab row of the $seat seat failed"
    else
      sj_ds41_ruler "$OUT/window$on_n/rows.json" "$OUT/window$off_n/rows.json" || sj_refuse 1 "the ab row of the $seat seat failed"
    fi
  done
}

# `bash tools/ref/sj-vs-clef.sh --self-test`: the stub servers (a tiny Python HTTP server that answers
# /v1/systemone, /v1/chat/completions and /tokenize with canned bodies), the windows, the servers stopped, the
# refusals and the ruler's math. No card, no lease, no model.
sj_selftest() {
  local n=0 fails=0 RC=0 C='' stub='' CP='' QP='' DP='' SELF=$HERE/sj-vs-clef.sh k p rows kv
  # T and RUNPID are read by the EXIT trap, after this function has returned: not local.
  RUNPID=''
  # The card pick and the dry flag are the cases' own, never the caller's.
  unset BLOOMERY_TIMING_GPU BLOOMERY_TIMING_CARDS BLOOMERY_BOX_CARD BLOOMERY_DRAFT BLOOMERY_OTHER_STRICT BLOOMERY_DRY \
    BLOOMERY_LEASE_CARD CUDA_VISIBLE_DEVICES
  command -v python3 > /dev/null || { echo "sj-vs-clef.sh --self-test: python3 is not on PATH" >&2; exit 2; }
  tmp=${TMPDIR:-/tmp}
  # The temporary root without a trailing slash, so the paths the cases expect are the paths printed.
  T=$(mktemp -d "${tmp%/}/sj-vs-clef-self-test.XXXXXX") || exit 2
  # st_reap <case dir>: KILL what the case's stub servers left, only a pid it logged whose command is still the stub.
  st_reap() {
    local p
    [ -r "$1/stub-pids" ] || return 0
    while read -r p; do
      if ps -o command= -p "$p" 2> /dev/null | grep -qF -- "$T/stub.py"; then kill -KILL "$p" 2> /dev/null; fi
    done < "$1/stub-pids"
    return 0
  }
  st_cleanup() {
    local p
    # The background case's shell and the runner it started, the pids the case wrote.
    p=$(cat "$T/case/runner.pid" 2> /dev/null) || p=''
    for p in "$RUNPID" $p; do
      [ -z "$p" ] || ! kill -0 "$p" 2> /dev/null || kill -KILL "$p" 2> /dev/null || :
    done
    st_reap "$T/case"
    rm -rf "$T"
  }
  trap st_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  # A host without timeout (the Mac) takes the stand-in the other self-tests use.
  mkdir "$T/bin"
  if ! command -v timeout > /dev/null; then
    cp "$HERE/stub-timeout.py" "$T/bin/timeout" && chmod +x "$T/bin/timeout" || exit 2
  fi
  PATH=$T/bin:$PATH
  ok() { n=$((n + 1)); echo "ok $1"; }
  bad() {
    n=$((n + 1)) fails=$((fails + 1))
    echo "FAIL $1: $2"
    [ ! -r "$C/out" ] || sed 's/^/    | /' "$C/out" | head -n 40
  }
  # The stub server: the endpoints the runner uses, with canned bodies.
  stub=$T/stub.py
  cat > "$stub" << 'STUB'
#!/usr/bin/env python3
"""A stub of bloomery-serve: the endpoints the runner uses, with canned bodies."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 0
MODEL = os.path.basename(os.environ.get("BLOOMERY_REF_MODEL", ""))
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])
    if a == "-m" and i + 1 < len(sys.argv):
        MODEL = os.path.basename(sys.argv[i + 1])
CONSTRAINED = os.environ.get("STUB_CONSTRAINED", "")
# A model file named plan* logs a plan record naming the A6000, other* one naming the 3090, as a Qwen seat's plan does;
# the Clef (decide) seat's log names no card.
CARD = {"plan": "A6000", "other": "3090"}.get(MODEL.split(".")[0], "")
SLOW = float(os.environ.get("STUB_SLOW", "0"))
# STUB_ROUTES: unset, /props answers 404; `none`, /props lists no routes; `yes`, it lists /v1/systemone.
ROUTES = os.environ.get("STUB_ROUTES", "")
LOG = os.path.join(os.environ["STUB_DIR"], "stub-requests") if os.environ.get("STUB_DIR") else ""
if MODEL.startswith("fail"):
    sys.exit(5)
if MODEL.startswith("hang"):
    time.sleep(3600)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def note(self, method):
        # The requests the bodies pass is ordered by, one line each in the order they came: port, method, path.
        if LOG:
            with open(LOG, "a") as f:
                f.write(f"{PORT} {method} {self.path}\n")

    def send_body(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/props":
            self.note("GET")
        if self.path in ("/health", "/v1/health"):
            self.send_body({"status": "ok"})
        elif self.path == "/props" and ROUTES:
            self.send_body({"build_info": "stub", **({"routes": ["/v1/systemone"]} if ROUTES == "yes" else {})})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/v1/systemone", "/v1/chat/completions"):
            self.note("POST")
        if self.path == "/tokenize":
            self.send_body({"tokens": [ord(c) for c in body.get("content", "")]})
        elif self.path == "/v1/systemone":
            answers = {}
            for qid, q in body.get("questions", {}).items():
                if q["type"] == "choice":
                    keys = sorted(q["criteria"])
                    answers[qid] = {"type": "choice", "choice": keys[0],
                                    "probabilities": {k: 0.5 for k in keys}}
                elif q["type"] == "score":
                    answers[qid] = {"type": "score",
                                    "probabilities": {str(i): 0.5 for i in range(len(q["criteria"]))}}
                else:
                    answers[qid] = {"type": "noul", "noul": 0.9}
            self.send_body({"model": "clef", "answers": answers,
                            "timings": {"prompt_n": 300, "prompt_ms": 70.0, "head_ms": 20.0}})
        elif self.path == "/v1/chat/completions":
            time.sleep(SLOW)
            props = body["response_format"]["json_schema"]["schema"]["properties"]
            out = {}
            for qid, prop in props.items():
                if prop["type"] == "string":
                    out[qid] = prop["enum"][0]
                elif prop["type"] == "integer":
                    out[qid] = prop["enum"][0]
                else:
                    out[qid] = True
            timings = {"prompt_n": 260, "prompt_ms": 50.0, "predicted_ms": 25.0}
            if CONSTRAINED == "bad":
                timings["constrained"] = {"arm": "jump", "spans": 2, "span_ids": 9, "decisions": 2,
                                          "free_steps": 0, "next_calls": 99, "prefill_calls": 3}
            self.send_body({"choices": [{"message": {"role": "assistant", "content": json.dumps(out)}}],
                            "timings": timings})
        else:
            self.send_error(404)


server = ThreadingHTTPServer(("127.0.0.1", PORT), H)
if CARD:
    print(f"plan place=a card={CARD} ctx_max=16384 devices=[stage:{CARD}:cuda0:50952404992] cuda_order=unset(FASTEST_FIRST)",
          file=sys.stderr, flush=True)
print(f"bloomery-serve-stub: arch=stub ctx=16384 slots=1 listening on http://127.0.0.1:{PORT}",
      file=sys.stderr, flush=True)
server.serve_forever()
STUB
  chmod +x "$stub"
  # The server binary the runner starts: the stub, behind a wrapper that logs the server's pid, the lever it was started
  # with (BLOOMERY_SCHEMAJUMP, `unset` when the environment has none) and the card it was given (CUDA_VISIBLE_DEVICES,
  # likewise) in the case's directory. Under STUB_CARD_GONE the card stops answering nvidia-smi from the first start on.
  cat > "$T/python3" << SERVE
#!/bin/sh
echo "\$\$" >> "\$STUB_DIR/stub-pids"
echo "\${BLOOMERY_SCHEMAJUMP-unset} \$*" >> "\$STUB_DIR/stub-levers"
echo "\${CUDA_VISIBLE_DEVICES-unset} \$*" >> "\$STUB_DIR/stub-cvd"
echo "\${BLOOMERY_REF_MODEL-unset}" >> "\$STUB_DIR/stub-refmodel"
[ -z "\${STUB_CARD_GONE:-}" ] || : > "\$STUB_DIR/smi-gone"
exec "$(command -v python3)" "$stub" "\$@"
SERVE
  chmod +x "$T/python3"
  # The stub nvidia-smi: the 3090 (device 0) and the A6000 (device 1) by name and UUID, the name of a card by its UUID,
  # and the compute processes of the A6000: STUB_BUSY_TIMING=<n> lists one for the first <n> polls of the run (the count
  # is a file of the case), of the 3090 the same under STUB_BUSY_OTHER. A UUID it does not list, or a card gone (smi-gone), fails as the driver does.
  cat > "$T/bin/nvidia-smi" << 'SMI'
#!/bin/sh
uuid='' prev=''
for a in "$@"; do
  [ "$prev" != -i ] || uuid=$a
  prev=$a
done
name_of() { case $1 in GPU-STUB-3090) echo 'NVIDIA GeForce RTX 3090' ;; GPU-STUB-A6000) echo 'NVIDIA RTX A6000' ;; *) return 1 ;; esac; }
case "$*" in
  *--query-gpu=name,uuid*) printf 'NVIDIA GeForce RTX 3090, GPU-STUB-3090\nNVIDIA RTX A6000, GPU-STUB-A6000\n'; exit 0 ;;
esac
if [ -n "$uuid" ]; then
  if ! name_of "$uuid" > /dev/null; then echo 'No devices were found' >&2; exit 6; fi
  if [ -e "$STUB_DIR/smi-gone" ]; then echo 'Unable to determine the device handle for GPU: Unknown Error' >&2; exit 15; fi
fi
case "$*" in
  *--query-compute-apps*)
    if [ "$uuid" = GPU-STUB-A6000 ] && [ "${STUB_BUSY_TIMING:-0}" != 0 ]; then
      n=$(cat "$STUB_DIR/smi-polls" 2> /dev/null || echo 0)
      echo $((n + 1)) > "$STUB_DIR/smi-polls"
      [ "$n" -ge "$STUB_BUSY_TIMING" ] || echo '4242, 1024 MiB'
    fi
    if [ "$uuid" = GPU-STUB-3090 ] && [ "${STUB_BUSY_OTHER:-0}" != 0 ]; then
      n=$(cat "$STUB_DIR/smi-polls-other" 2> /dev/null || echo 0)
      echo $((n + 1)) > "$STUB_DIR/smi-polls-other"
      [ "$n" -ge "$STUB_BUSY_OTHER" ] || echo '5151, 2048 MiB'
    fi
    ;;
  *--query-gpu=name*) name_of "$uuid" ;;
esac
exit 0
SMI
  chmod +x "$T/bin/nvidia-smi"
  : > "$T/model.gguf"
  : > "$T/fail.gguf"
  : > "$T/hang.gguf"
  : > "$T/plan.gguf"
  : > "$T/other.gguf"
  st_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
  # st_prep: a fresh case directory and the two ports; the previous case's servers reaped first.
  st_prep() {
    st_reap "$T/case"
    C=$T/case
    rm -rf "$C"
    mkdir -p "$C"
    : > "$C/out"
    # The tree the runner stands in (assert_fresh_binary walks it): sources older than the binary, or, under ST_STALE,
    # one of them newer.
    mkdir -p "$C/tree/crates/x/src"
    : > "$C/tree/Cargo.toml"
    : > "$C/tree/Cargo.lock"
    echo 'fn main() {}' > "$C/tree/crates/x/src/lib.rs"
    touch -t 200001010000 "$C/tree/Cargo.toml" "$C/tree/Cargo.lock" "$C/tree/crates/x/src/lib.rs"
    [ -z "${ST_STALE:-}" ] || touch -t 203001010000 "$C/tree/crates/x/src/lib.rs"
    CP=$(st_port)
    QP=$(st_port)
    DP=$(st_port)
  }
  # st_launch <function and arguments>: the runner's function in a bash of its own, as the justfile runs the script, so
  # its $$ and its traps are the runner's; its output in $C/out. The lease, the witness and the guards are stand-ins.
  # ST_ENV holds the knobs and the stub's.
  ST_ENV=()
  st_launch() {
    (
      export STUB_DIR=$C SJ_SERVE_BIN=$T/python3 SJ_OUT=$C/out-dir SJ_LOGDIR=$C
      export SJ_CLEF_PORT=$CP SJ_QWEN_PORT=$QP SJ_DS41_PORT=$DP SJ_DS41_BIN=$T/python3 SJ_ROUNDS=2 SJ_ROUNDS_LONG=1 SJ_SERVER_BOUND=60 SJ_RUN_BOUND=30
      # shellcheck disable=SC2163 # kv is a NAME=VALUE string
      for kv in ${ST_ENV[@]+"${ST_ENV[@]}"}; do export "$kv"; done
      cd "$C/tree" || exit 2
      exec bash -c 'echo $$ > "$STUB_DIR/runner.pid"; source "$1"; shift
        lease_take() { echo "[lease-stub] taken"; }
        witness() { echo "[witness-stub] $1"; }
        guard_cpu() { :; }
        SJ_BUSY_POLLS=${ST_POLLS:-3} SJ_BUSY_STEP=1
        "$@"' _ "$SELF" "$@"
    ) > "$C/out" 2>&1
  }
  # st_run: the runner, to its end; its rc in RC.
  st_run() {
    st_prep
    st_launch sj_main
    RC=$?
  }
  st_check() {
    local name=$1 want=$2 s
    shift 2
    if [ "$RC" != "$want" ]; then bad "$name" "rc $RC, want $want"; return; fi
    for s in "$@"; do
      grep -qF -- "$s" "$C/out" || { bad "$name" "no line holds '$s'"; return; }
    done
    ok "$name"
  }
  st_absent() {
    local name=$1 s
    shift
    for s in "$@"; do
      if grep -qF -- "$s" "$C/out"; then bad "$name" "a line holds '$s'"; return; fi
    done
    ok "$name"
  }
  # st_servers <name> <count>: the stub server started <count> times in the case and none of them lives.
  st_servers() {
    local p alive=0 count=0
    if [ -r "$C/stub-pids" ]; then
      while read -r p; do
        count=$((count + 1))
        ! kill -0 "$p" 2> /dev/null || alive=$((alive + 1))
      done < "$C/stub-pids"
    fi
    if [ "$count" != "$2" ]; then bad "$1" "$count servers started, want $2"; return; fi
    if [ "$alive" != 0 ]; then bad "$1" "$alive of $count servers still up after the run"; return; fi
    ok "$1"
  }
  # st_levers: who was started with which lever: `clef=unset qwen=on ...`, in the order they started.
  st_levers() {
    awk -v c="$CP" '{ for (i = 1; i <= NF; i++) if ($i == "--port") printf "%s=%s ", ($(i + 1) == c ? "clef" : "qwen"), $1 }' \
      "$C/stub-levers"
  }
  # st_bodies <name> <window dir> <arm>...: each arm's first counted round's answer of every request is a file the row
  # names (the dropped round's row names none, no other row does), and the directory holds those files and no more.
  st_bodies() {
    local name=$1 dir=$2 msg
    shift 2
    if msg=$(python3 -I - "$dir" "$@" 2>&1 << 'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = json.loads((root / "rows.json").read_text())["rows"]
for arm in sys.argv[2:]:
    mine = [r for r in rows if r["arm"] == arm]
    keys = sorted({(r["suite"], r["id"]) for r in mine})
    for suite, rid in keys:
        rs = [r for r in mine if r["suite"] == suite and r["id"] == rid]
        named = [r for r in rs if "body" in r]
        first = min(r["round"] for r in rs if not r["dropped"])
        if len(named) != 1 or named[0]["dropped"] or named[0]["round"] != first:
            sys.exit(f"{arm} {suite}/{rid}: the rows naming a body are {[(r['round'], r['dropped']) for r in named]}, want round {first}")
        if named[0]["body"] != f"bodies/{arm}/{suite}-{rid}.json":
            sys.exit(f"{arm} {suite}/{rid}: the row names {named[0]['body']}")
        json.loads((root / named[0]["body"]).read_text())
    files = sorted(p.name for p in (root / "bodies" / arm).iterdir())
    if files != sorted(f"{s}-{i}.json" for s, i in keys):
        sys.exit(f"{arm}: the directory holds {len(files)} files, want one per request ({len(keys)})")
print(f"{len(sys.argv) - 2} arms, {len(keys)} requests each")
PY
    ); then ok "$name"; else bad "$name" "$msg"; fi
  }
  # st_systemone <name> <window dir> <arm> <n>: the arm's SystemOne bodies are <n> files, each a SystemOne answer.
  st_systemone() {
    local name=$1 dir=$2 msg
    if msg=$(python3 -I - "$dir" "$3" "$4" 2>&1 << 'PY'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) / "bodies" / f"{sys.argv[2]}-systemone"
files = sorted(d.glob("*.json")) if d.is_dir() else []
if len(files) != int(sys.argv[3]):
    sys.exit(f"{d} holds {len(files)} bodies, want {sys.argv[3]}")
for f in files:
    if "answers" not in json.loads(f.read_text()):
        sys.exit(f"{f.name} is no SystemOne answer")
if list(d.glob("*.status")):
    sys.exit("a 200 answer has a status file")
PY
    ); then ok "$name"; else bad "$name" "$msg"; fi
  }
  # st_order <name> <qwen pattern>: the requests of the stub servers in the order they came (the stub's log, one token a
  # request: C a chat POST, S a SystemOne POST, G a GET /props): the Clef server's are all S and none G, the generative
  # server's match the pattern (a regular expression over those tokens).
  st_order() {
    local name=$1 msg
    if msg=$(python3 -I - "$C/stub-requests" "$CP" "$QP" "$2" 2>&1 << 'PY'
import re
import sys

log, clef, qwen, pattern = sys.argv[1:5]
tok = {"POST /v1/chat/completions": "C", "POST /v1/systemone": "S", "GET /props": "G"}
seen = {clef: "", qwen: ""}
for line in open(log):
    port, rest = line.rstrip("\n").split(" ", 1)
    seen[port] += tok[rest]
if not re.fullmatch("S+", seen[clef]):
    sys.exit(f"the Clef server saw {seen[clef]}, want S only")
if not re.fullmatch(pattern, seen[qwen]):
    sys.exit(f"the generative server saw {seen[qwen]}, want {pattern}")
PY
    ); then ok "$name"; else bad "$name" "$msg"; fi
  }
  MODELS=("SJ_CLEF_MODEL=$T/model.gguf" "SJ_QWEN_MODEL=$T/model.gguf")

  # --- the refusals before the lease.
  ST_ENV=()
  st_run
  st_check refuse-no-clef-model 64 "SJ_CLEF_MODEL names the Clef model file"
  ST_ENV=("SJ_CLEF_MODEL=$T/model.gguf")
  st_run
  st_check refuse-no-qwen-model 64 "SJ_QWEN_MODEL names the Qwen3.6 model file"
  ST_ENV=("SJ_CLEF_MODEL=$T/nosuch.gguf" "SJ_QWEN_MODEL=$T/model.gguf")
  st_run
  st_check refuse-missing-model 2 "no Clef model file at"
  ST_ENV=("${MODELS[@]}" "SJ_CLEF_PORT=70000")
  st_run
  st_check refuse-port 64 "SJ_CLEF_PORT is a port"
  ST_ENV=("${MODELS[@]}" "SJ_ROUNDS=x")
  st_run
  st_check refuse-rounds 64 "ROUNDS is a whole number"
  ST_ENV=("${MODELS[@]}" "SJ_SERVER_BOUND=x")
  st_run
  st_check refuse-server-bound 64 "SJ_SERVER_BOUND is a whole number"
  ST_ENV=("${MODELS[@]}" "SJ_SERVER_BOUND=10" "SJ_RUN_BOUND=10")
  st_run
  st_check refuse-run-bound 64 "SJ_RUN_BOUND 10 s is inside SJ_SERVER_BOUND 10 s"
  # A BLOOMERY_SERVE_ name is a prefix, not a name (the check-levers rule): the name is spelled in two
  # pieces here, so the tree holds no name of its own.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_SERVE_"CTX=1)
  st_run
  st_check refuse-prefixed-knob 64 "is set, and nothing reads it"
  st_absent refuse-before-the-lease "[lease-stub]"

  # --- a dry run: the plan, both windows' servers with the lever, nothing started and no lease.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1")
  st_run
  st_check dry-plan 0 "[plan] bounds: server 60s" "[plan] window 1 qwen: timeout 60 env -u BLOOMERY_REF_MODEL BLOOMERY_SCHEMAJUMP=on " \
    "[plan] window 2 qwen: timeout 60 env -u BLOOMERY_REF_MODEL BLOOMERY_SCHEMAJUMP=off " \
    "[plan] window 2 clef: timeout 60 env -u BLOOMERY_REF_MODEL -u BLOOMERY_SCHEMAJUMP " "[dry] nothing built, no lease taken, no server started"
  st_absent dry-takes-no-lease "[lease-stub]" "[witness-stub]"
  [ ! -e "$C/stub-pids" ] && ok dry-starts-no-server || bad dry-starts-no-server "the stub was started"
  st_absent dry-with-a-binary-no-note "no server binary"
  # A tree with no binary yet: the plan with a note under a dry run, a refusal (2) under a real one.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1" "SJ_SERVE_BIN=$T/nosuch/bloomery-serve")
  st_run
  st_check dry-missing-binary 0 "[dry] no server binary at $T/nosuch/bloomery-serve: a real run is refused (rc 2)" "[plan] window 1 clef:"
  ST_ENV=("${MODELS[@]}" "SJ_SERVE_BIN=$T/nosuch/bloomery-serve")
  st_run
  st_check refuse-missing-binary 2 "no server binary at $T/nosuch/bloomery-serve (the recipe builds it)"

  # --- a run: the two windows, the servers started and stopped, the rows, the tables.
  ST_ENV=("${MODELS[@]}")
  st_run
  st_check run 0 "[lease-stub] taken" "[witness-stub] pre" "[witness-stub] post" \
    "[window 1] the Clef arm and the jump arm" "[window 2] the Clef arm and the step arm" \
    "diff (jump - step) / step"
  st_check run-tables 0 "=== per arm: the client wall" "=== agreement ==="
  rows=$(ls "$C/out-dir/window1/rows.json" "$C/out-dir/window2/rows.json" 2> /dev/null | wc -l | tr -d ' ')
  [ "$rows" = 2 ] && ok run-rows || bad run-rows "the rows are $rows files, want 2"
  st_servers run-servers-stopped 4
  # The lever: on in window 1, off in window 2, on the Qwen3.6 server only.
  [ "$(st_levers)" = "clef=unset qwen=on clef=unset qwen=off " ] && ok run-levers || bad run-levers "got [$(st_levers)]"
  # The bodies. The stub answers /props with 404, so each generative arm's SystemOne pass is skipped by name; the timed
  # bodies are there for both arms of the window (2 arms x 15 requests).
  st_check run-bodies-skipped 0 "[bodies] timed: 30 bodies of 2 arms under $C/out-dir/window1/bodies" \
    "[bodies] jump: the server lists no /v1/systemone; skipped (GET /props answered 404)" \
    "[bodies] step: the server lists no /v1/systemone; skipped (GET /props answered 404)"
  st_bodies run-bodies-window1 "$C/out-dir/window1" A jump
  st_bodies run-bodies-window2 "$C/out-dir/window2" A step
  [ ! -e "$C/out-dir/window1/bodies/jump-systemone" ] && ok run-bodies-no-systemone-dir || bad run-bodies-no-systemone-dir "a directory of SystemOne bodies stands"
  # The server lists the route: after the rounds each generative arm is asked for /props once and for each request's
  # SystemOne body once, nothing else comes after; the Clef arm is never asked for /props and has no pass.
  ST_ENV=("${MODELS[@]}" "STUB_ROUTES=yes")
  st_run
  st_check run-systemone-pass 0 "[bodies] jump: systemone 15 ok / 15" "[bodies] step: systemone 15 ok / 15"
  st_absent run-systemone-skips-none "the server lists no /v1/systemone"
  st_systemone run-systemone-bodies-jump "$C/out-dir/window1" jump 15
  st_systemone run-systemone-bodies-step "$C/out-dir/window2" step 15
  st_bodies run-systemone-timed-bodies "$C/out-dir/window1" A jump
  [ ! -e "$C/out-dir/window1/bodies/A-systemone" ] && ok run-systemone-clef-has-none || bad run-systemone-clef-has-none "the Clef arm has a pass"
  st_order run-systemone-order 'C+GS{15}C+GS{15}'
  # A /props that lists no routes: skipped by name (no suffix), no POST of the route.
  ST_ENV=("${MODELS[@]}" "STUB_ROUTES=none" "SJ_WINDOWS=qwen:on")
  st_run
  st_check run-no-routes 0 "[bodies] jump: the server lists no /v1/systemone; skipped"
  grep -qxF -- "[bodies] jump: the server lists no /v1/systemone; skipped" "$C/out" && ok run-no-routes-no-suffix ||
    bad run-no-routes-no-suffix "no line is exactly the skipped line"
  st_order run-no-routes-order 'C+G'
  # A value in the caller's environment picks no arm.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_SCHEMAJUMP=off")
  st_run
  st_check run-inherited-lever 0 "[window 2] the Clef arm and the step arm"
  [ "$(st_levers)" = "clef=unset qwen=on clef=unset qwen=off " ] && ok run-inherited-levers || bad run-inherited-levers "got [$(st_levers)]"

  # --- the timing card (the stub nvidia-smi lists the 3090 as device 0 and the A6000 as device 1; the A6000 is the
  # default timing card). st_cvds: the CUDA_VISIBLE_DEVICES each server started with, in the order they started.
  st_cvds() { awk '{ printf "%s ", $1 }' "$C/stub-cvd"; }
  # The pin: a caller's CUDA_VISIBLE_DEVICES (the 3090's) reaches no server; all four got the A6000's.
  ST_ENV=("${MODELS[@]}" "CUDA_VISIBLE_DEVICES=GPU-STUB-3090")
  st_run
  st_check card-pin-run 0 "[window 2] the Clef arm and the step arm"
  [ "$(st_cvds)" = "GPU-STUB-A6000 GPU-STUB-A6000 GPU-STUB-A6000 GPU-STUB-A6000 " ] && ok card-pin-servers || bad card-pin-servers "got [$(st_cvds)]"
  # The config line and the servers' log names, the card's witness field among the runner's.
  st_check card-config-line 0 "[config] timing_gpu=GPU-STUB-A6000 other_gpu=GPU-STUB-3090 CUDA_VISIBLE_DEVICES=GPU-STUB-A6000 "
  # A server whose log names another card: the window is refused by name and both servers are stopped.
  ST_ENV=("SJ_CLEF_MODEL=$T/model.gguf" "SJ_QWEN_MODEL=$T/other.gguf")
  st_run
  st_check card-log-other 1 "window 1: the qwen server's log names the card 3090, and the timing card is NVIDIA RTX A6000"
  st_absent card-log-other-no-table "=== per arm" "[window 1] the Clef arm"
  st_servers card-log-other-servers-gone 2
  # A log that names the timing card passes, and a log that names none (the Clef seat's) is told apart in a line.
  ST_ENV=("SJ_CLEF_MODEL=$T/model.gguf" "SJ_QWEN_MODEL=$T/plan.gguf")
  st_run
  st_check card-log-named 0 "[card] window 1 qwen: the log names A6000, the timing card (NVIDIA RTX A6000)" \
    "[card] window 1 clef: the log names no card; CUDA_VISIBLE_DEVICES=GPU-STUB-A6000 pins it"
  # A card that stops answering after a window is refused by name, and the servers are stopped.
  ST_ENV=("${MODELS[@]}" "STUB_CARD_GONE=1")
  st_run
  st_check card-gone 1 "window 1: the timing card (GPU-STUB-A6000) does not answer nvidia-smi (rc 15)"
  st_servers card-gone-servers-gone 2
  # A card pick that does not answer is refused before the lease.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_TIMING_GPU=GPU-STUB-NOSUCH")
  st_run
  st_check card-unknown 69 "refused before the lease: the timing card (GPU-STUB-NOSUCH) does not answer nvidia-smi (rc 6)"
  st_absent card-unknown-before-the-lease "[lease-stub]"
  # The timing card busy with another process: the window waits for it, and ends 75 when the wait runs out, with no
  # server started; a card freed within the wait lets the window go on.
  ST_ENV=("${MODELS[@]}" "STUB_BUSY_TIMING=999" "ST_POLLS=3")
  st_run
  st_check card-busy 75 "[timing-busy]" "compute apps on the timing card (GPU-STUB-A6000)" "still busy after 3 s"
  [ ! -e "$C/stub-pids" ] && ok card-busy-starts-no-server || bad card-busy-starts-no-server "a server was started"
  ST_ENV=("${MODELS[@]}" "STUB_BUSY_TIMING=2" "ST_POLLS=5")
  st_run
  st_check card-busy-then-free 0 "[timing-busy]" "the timing card is free after 2 s" "[window 2] the Clef arm and the step arm"
  # A binary older than its sources is refused (3) before the lease, and a dry run says so and goes on.
  ST_STALE=1
  ST_ENV=("${MODELS[@]}")
  st_run
  st_check refuse-stale-binary 3 "[stale-binary] $T/python3" "is older than its sources"
  st_absent refuse-stale-binary-before-the-lease "[lease-stub]"
  [ ! -e "$C/stub-pids" ] && ok refuse-stale-binary-starts-no-server || bad refuse-stale-binary-starts-no-server "a server was started"
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1")
  st_run
  st_check dry-stale-binary 0 "[dry] $T/python3 would be refused (rc 3)" "[plan] window 1 clef:"
  ST_STALE=''
  # A dry run prints the card pick, and each server line carries it.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1")
  st_run
  st_check dry-card-pick 0 "[plan] card: timing_gpu=GPU-STUB-A6000 (NVIDIA RTX A6000) other_gpu=GPU-STUB-3090 CUDA_VISIBLE_DEVICES=GPU-STUB-A6000" \
    "[plan] window 1 clef: timeout 60 env -u BLOOMERY_REF_MODEL -u BLOOMERY_SCHEMAJUMP CUDA_VISIBLE_DEVICES=GPU-STUB-A6000 " \
    "[plan] window 2 qwen: timeout 60 env -u BLOOMERY_REF_MODEL BLOOMERY_SCHEMAJUMP=off CUDA_VISIBLE_DEVICES=GPU-STUB-A6000 "


  # --- the suites by name, the gold, the windows and the V4.1 seat.
  mkdir -p "$T/suites"
  python3 - "$T/suites" << 'FIX'
import json
import sys

rows = [
    {"id": "f-1", "state": "A customer wants the money back.",
     "questions": {"intent": {"type": "choice", "criteria": {"refund": "R", "cancel": "C", "other": "O"}},
                   "angry": {"type": "noul"}}},
    {"id": "f-2", "state": "Fine.", "questions": {"level": {"type": "score", "criteria": ["low", "mid", "high"]}}},
]
gold = [{"id": "f-1", "gold": {"intent": "cancel", "angry": "yes"}, "why": "x"},
        {"id": "f-2", "gold": {"level": "1"}, "why": "y"}]
for name, g in (("suite-fix", gold), ("suite-badgold", gold + [{"id": "f-9", "gold": {"q": "a"}}])):
    open(f"{sys.argv[1]}/{name}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    open(f"{sys.argv[1]}/{name}.gold.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in g))
FIX
  FIXSUITE=$T/suites/suite-fix.jsonl
  # The default list stays, and the plan names the suites read.
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1")
  st_run
  st_check suites-default-plan 0 "[plan] windows: qwen:on qwen:off suites: tools/ref/clef/suite.jsonl tools/ref/clef/suite-ko.jsonl" \
    "[suites] en (tools/ref/clef/suite.jsonl)" "[suites] ko (tools/ref/clef/suite-ko.jsonl)"
  ST_ENV=("${MODELS[@]}" "BLOOMERY_DRY=1" "SJ_SUITES=$FIXSUITE")
  st_run
  st_check suites-named-plan 0 "suites: $FIXSUITE" "[suites] fix ($FIXSUITE): 2 requests, 3 questions, gold of 3 questions"
  # A suite or a gold file that does not load ends the run before the lease, dry or not, by name.
  ST_ENV=("${MODELS[@]}" "SJ_SUITES=$T/suites/suite-badgold.jsonl")
  st_run
  st_check refuse-bad-gold 64 "the suites do not load" "the gold names the id 'f-9', which the suite badgold does not hold"
  st_absent refuse-bad-gold-before-the-lease "[lease-stub]"
  ST_ENV=("${MODELS[@]}" "SJ_SUITES=$T/suites/nosuch.jsonl")
  st_run
  st_check refuse-missing-suite 64 "the suites do not load" "no suite file at $T/suites/nosuch.jsonl"
  ST_ENV=("${MODELS[@]}" "SJ_SUITES= ")
  st_run
  st_check refuse-no-suite 64 "SJ_SUITES names no suite file"
  # A named suite with its gold: every arm is scored (the stubs pick the first option of every field; the gold of f-1 is
  # `cancel` and `yes`, the first options, and f-2's level 1 is not the stubs' 0).
  ST_ENV=("${MODELS[@]}" "SJ_SUITES=$FIXSUITE")
  st_run
  st_check suites-gold-run 0 "=== gold: the picks against the labels ===" "arm A (clef) vs the gold of suite fix | 2/3" \
    "arm jump (qwen) vs the gold of suite fix | 2/3" "arm step (qwen) vs the gold of suite fix | 2/3" "suite fix" "correct 2/3" \
    "wrong: fix/f-2/level (gold 1, ours 0)"
  st_absent suites-gold-needs-no-official-rows "the official rows are not at"

  # --- the windows: a lone window has no lever pair row.
  ST_ENV=("${MODELS[@]}" "SJ_SUITES=$FIXSUITE" "SJ_WINDOWS=qwen:on")
  st_run
  st_check windows-lone 0 "[window 1] the Clef arm and the jump arm: " "arm jump (qwen) | suite fix"
  st_absent windows-lone-no-lever-pair "diff (jump - step) / step"
  st_servers windows-lone-servers-stopped 2
  # The lever pair is found in whichever windows it stands.
  ST_ENV=("${MODELS[@]}" "SJ_WINDOWS=qwen:off qwen:on")
  st_run
  st_check windows-reversed 0 "[window 1] the Clef arm and the step arm" "[window 2] the Clef arm and the jump arm" "diff (jump - step) / step"
  [ "$(st_levers)" = "clef=unset qwen=off clef=unset qwen=on " ] && ok windows-reversed-levers || bad windows-reversed-levers "got [$(st_levers)]"
  # The words may be split by commas, as a sitting passes them (BLOOMERY_BOX_ENV is split on spaces).
  ST_ENV=("${MODELS[@]}" "SJ_DS41_MODEL=$T/model.gguf" "BLOOMERY_DRY=1" "SJ_WINDOWS=qwen:on,ds41:off" "SJ_SUITES=$FIXSUITE,tools/ref/clef/suite-upstream.jsonl")
  st_run
  st_check windows-suites-by-commas 0 "[plan] windows: qwen:on ds41:off suites: $FIXSUITE tools/ref/clef/suite-upstream.jsonl" \
    "[suites] fix (" "[suites] upstream (tools/ref/clef/suite-upstream.jsonl)" "[plan] window 2 ds41:"
  # A window word that is not one is refused by name, before the lease.
  for kv in "foo:on|names the seat 'foo'" "qwen:maybe|names the lever 'maybe'" "qwen|is <seat>:<on|off>" \
    "qwen:on+chat|names the lever 'on+chat'" "qwen:on qwen:on|names qwen:on twice" " |SJ_WINDOWS names no window"; do
    ST_ENV=("${MODELS[@]}" "SJ_WINDOWS=${kv%%|*}")
    st_run
    st_check "refuse-window-[${kv%%|*}]" 64 "${kv#*|}"
  done
  st_absent refuse-window-before-the-lease "[lease-stub]"

  # --- the V4.1 seat: one server, no Clef, both cards visible with the timing card first, the model by the environment.
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:on" "SJ_SUITES=$FIXSUITE")
  st_run
  st_check ds41-window 0 "[window 1] the ds41-jump arm: " "[card] window 1 ds41: the log names no card; the server was given CUDA_VISIBLE_DEVICES=GPU-STUB-A6000,GPU-STUB-3090" \
    "arm ds41-jump (qwen) | suite fix" "arm ds41-jump (qwen) vs the gold of suite fix | 2/3"
  st_absent ds41-window-no-clef "the Clef arm" "diff (jump - step)" "chat"

  st_servers ds41-window-servers-stopped 1
  [ "$(awk '{ print $1 }' "$C/stub-levers")" = on ] && ok ds41-window-lever || bad ds41-window-lever "got [$(cat "$C/stub-levers")]"
  [ "$(cat "$C/stub-cvd" | awk '{ print $1 }')" = "GPU-STUB-A6000,GPU-STUB-3090" ] && ok ds41-window-cards || bad ds41-window-cards "got [$(cat "$C/stub-cvd")]"
  [ "$(cat "$C/stub-refmodel")" = "$T/model.gguf" ] && ok ds41-window-model-by-env || bad ds41-window-model-by-env "got [$(cat "$C/stub-refmodel")]"
  grep -q -- '--place' "$C/stub-levers" && bad ds41-window-no-place "the server was given --place" || ok ds41-window-no-place
  ST_ENV=("SJ_WINDOWS=ds41:on")
  st_run
  st_check refuse-no-ds41-model 64 "SJ_DS41_MODEL names the V4.1 model file"
  st_absent refuse-no-ds41-model-asks-no-clef "SJ_CLEF_MODEL"
  ST_ENV=("SJ_WINDOWS=ds41:on" "SJ_DS41_MODEL=$T/nosuch.gguf")
  st_run
  st_check refuse-missing-ds41-model 2 "no V4.1 model file at $T/nosuch.gguf"
  ST_ENV=("SJ_WINDOWS=ds41:on ds41:off" "SJ_DS41_MODEL=$T/model.gguf" "BLOOMERY_DRY=1")
  st_run
  st_check dry-ds41-plan 0 "[plan] window 1 ds41: timeout 60 env BLOOMERY_REF_MODEL=$T/model.gguf BLOOMERY_SCHEMAJUMP=on CUDA_VISIBLE_DEVICES=GPU-STUB-A6000,GPU-STUB-3090 $T/python3 --host 127.0.0.1 --port " \
    "[plan] window 2 ds41: timeout 60 env BLOOMERY_REF_MODEL=$T/model.gguf BLOOMERY_SCHEMAJUMP=off " "[plan] window 1 arms: ds41:on" "[plan] window 2 arms: ds41:off"
  st_absent dry-ds41-plan-no-clef "window 1 clef"
  # The V4.1 seat's two arms are the same pair the Qwen seat runs, both through response_format: the jump arm against
  # the step arm, the answers held equal (SAME), the wall's ab row beside them.
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:off,ds41:on" "SJ_SUITES=$FIXSUITE")
  st_run
  st_check ds41-pair 0 "[window 1] the ds41-step arm: " "[window 2] the ds41-jump arm: " \
    "arm ds41-jump (qwen) vs the gold of suite fix | 2/3" "arm ds41-step (qwen) vs the gold of suite fix | 2/3" \
    "ds41-jump vs ds41-step | SAME 2/2" "=== the ds41 jump arm against the ds41 step arm (the short bucket's wall p50, paired by round) ===" \
    "diff (ds41 jump - ds41 step) / ds41 step"
  st_absent ds41-pair-no-qwen-pair "diff (jump - step) / step" "the card's ruler"
  st_servers ds41-pair-servers-stopped 2
  [ "$(awk '{ print $1 }' "$C/stub-levers" | tr '\n' ' ')" = "off on " ] && ok ds41-pair-levers || bad ds41-pair-levers "got [$(cat "$C/stub-levers")]"
  # The hard sitting's shape: Clef and the Qwen jump arm in one window, the V4.1 pair after, each arm scored on its own.
  ST_ENV=("${MODELS[@]}" "SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=qwen:on,ds41:on,ds41:off" "SJ_SUITES=$FIXSUITE")
  st_run
  st_check hard-shape 0 "[window 1] the Clef arm and the jump arm: " "[window 2] the ds41-jump arm: " "[window 3] the ds41-step arm: " \
    "arm A (clef) vs the gold of suite fix | 2/3" "arm jump (qwen) vs the gold of suite fix | 2/3" \
    "arm ds41-jump (qwen) vs the gold of suite fix | 2/3" "arm ds41-step (qwen) vs the gold of suite fix | 2/3" "ds41-jump vs ds41-step | SAME 2/2" \
    "diff (ds41 jump - ds41 step) / ds41 step"
  st_absent hard-shape-no-qwen-pair "diff (jump - step) / step"
  st_servers hard-shape-servers-stopped 4
  # It starts on neither card while another process holds one: the guard waits, then ends the run at 75 naming the card
  # and the process, with no server started. A process on the 3090 does not stop a single-card seat.
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:on" "STUB_BUSY_OTHER=999" "ST_POLLS=3")
  st_run
  st_check ds41-refuses-held-3090 75 "[timing-busy]" "compute apps on the other card (GPU-STUB-3090): [5151, 2048 MiB;]" "still busy after 3 s" \
    "the V4.1 server needs both cards and starts on neither while another process holds one"
  [ ! -e "$C/stub-pids" ] && ok ds41-refuses-held-3090-starts-no-server || bad ds41-refuses-held-3090-starts-no-server "a server was started"
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:on" "STUB_BUSY_TIMING=999" "ST_POLLS=3")
  st_run
  st_check ds41-refuses-held-a6000 75 "compute apps on the timing card (GPU-STUB-A6000): [4242, 1024 MiB;]" "still busy after 3 s"
  [ ! -e "$C/stub-pids" ] && ok ds41-refuses-held-a6000-starts-no-server || bad ds41-refuses-held-a6000-starts-no-server "a server was started"
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:on" "STUB_BUSY_TIMING=999" "STUB_BUSY_OTHER=999" "ST_POLLS=2")
  st_run
  st_check ds41-refuses-both-held 75 "compute apps on the timing card (GPU-STUB-A6000): [4242, 1024 MiB;]; compute apps on the other card (GPU-STUB-3090): [5151, 2048 MiB;]"
  ST_ENV=("SJ_DS41_MODEL=$T/model.gguf" "SJ_WINDOWS=ds41:on" "SJ_SUITES=$FIXSUITE" "STUB_BUSY_OTHER=2" "ST_POLLS=5")
  st_run
  # (the runner's own guard_other read took the first of the two polls)
  st_check ds41-held-then-free 0 "[timing-busy]" "the timing card is free after 1 s" "[window 1] the ds41-jump arm"
  ST_ENV=("${MODELS[@]}" "SJ_SUITES=$FIXSUITE" "STUB_BUSY_OTHER=999")
  st_run
  st_check qwen-runs-beside-a-held-3090 0 "[window 1] the Clef arm and the jump arm" "[other-busy]"

  # --- a server that does not start: the other is gone when the runner ends, and a refusal names why.
  ST_ENV=("SJ_CLEF_MODEL=$T/model.gguf" "SJ_QWEN_MODEL=$T/fail.gguf")
  st_run
  st_check refuse-second-server 1 "the qwen server: llama-server exited 5 before it answered /health"
  st_servers refuse-second-server-first-gone 2
  # A server that never listens is ended at its bound, and the first with it.
  ST_ENV=("SJ_CLEF_MODEL=$T/model.gguf" "SJ_QWEN_MODEL=$T/hang.gguf" "SJ_SERVER_BOUND=5" "SJ_RUN_BOUND=2")
  st_run
  st_check refuse-hung-server 1 "the qwen server: llama-server did not answer /health within 5 s"
  st_servers refuse-hung-server-gone 2

  # --- a TERM mid-window: the run, both servers and the lease go with the runner, which ends 143.
  ST_ENV=("${MODELS[@]}" "STUB_SLOW=30")
  st_prep
  ST_ENV+=("LEASE_BOUNDED_PIDFILE=$C/bounded.pid")
  st_launch sj_main &
  RUNPID=$!
  k=0
  # The pidfile holds the last lease_bounded command's pid (the card queries and the suites' check before the window are
  # bounded too): wait for the one that is the request run.
  st_run_bounded() { ps -o command= -p "$(cat "$C/bounded.pid" 2> /dev/null)" 2> /dev/null | grep -qF 'sjcmp.py run'; }
  until { [ -s "$C/runner.pid" ] && st_run_bounded; } || [ "$k" = 300 ]; do sleep 0.1; k=$((k + 1)); done
  st_run_bounded && ok term-run-is-bounded || bad term-run-is-bounded "no request run went through lease_bounded in 30 s"
  p=$(cat "$C/runner.pid" 2> /dev/null) || p=''
  [ -z "$p" ] || kill -TERM "$p" 2> /dev/null
  RC=0
  { wait "$RUNPID" || RC=$?; } 2> /dev/null # the shell's "Terminated" notice
  RUNPID=''
  st_check term-ends-143 143
  st_servers term-servers-gone 2
  p=$(cat "$C/bounded.pid" 2> /dev/null) || p=''
  if [ -n "$p" ] && kill -0 "$p" 2> /dev/null; then bad term-run-gone "the bounded run (pid $p) still lives"; else ok term-run-gone; fi

  # --- the EXIT trap leaves lease.sh's own cleanup (the capture directory lease_out_init appends to the trap) standing.
  st_prep
  RC=0
  bash -c 'source "$1"; trap sj_cleanup EXIT; lease_out_init || exit 70; echo "$LEASE_OUT_DIR" > "$2"; exit 5' _ "$SELF" "$C/capture-dir" \
    > "$C/out" 2>&1 || RC=$?
  p=$(cat "$C/capture-dir" 2> /dev/null) || p=''
  if [ "$RC" = 5 ] && [ -n "$p" ] && [ ! -e "$p" ]; then ok exit-trap-keeps-lease-cleanup; else bad exit-trap-keeps-lease-cleanup "rc $RC, capture directory '$p' $([ -e "$p" ] && echo left || echo gone)"; fi

  # --- a count mismatch stops the run by name.
  ST_ENV=("${MODELS[@]}" "SJ_ROUNDS=1" "STUB_CONSTRAINED=bad")
  st_run
  st_check refuse-count-mismatch 1 "next_calls 99"
  st_absent refuse-count-mismatch-no-table "=== per arm"
  st_servers refuse-count-mismatch-servers-gone 2

  # --- the ruler's math, against card.py's paired_half_width.
  st_prep
  mkdir -p "$C/window1" "$C/window2"
  python3 - "$C/window1/rows.json" "$C/window2/rows.json" << 'ROWS'
import json
import sys


def rows(path, arm, base):
    out = []
    for r in range(1, 5):
        for i in range(3):
            out.append({"arm": arm, "kind": "qwen", "suite": "en", "id": f"r{i}", "round": r,
                        "dropped": False, "wall_ms": base, "server_ms": base, "host_ms": 0.0,
                        "bucket": "short", "tops": {"q": "a"}})
    return {"rows": out}


json.dump(rows(sys.argv[1], "B", 100.0), open(sys.argv[1], "w"))
json.dump(rows(sys.argv[2], "C", 125.0), open(sys.argv[2], "w"))
ROWS
  RC=0
  bash -c 'source "$1"; OUT=$2; sj_ruler' _ "$SELF" "$C" > "$C/out" 2>&1 || RC=$?
  st_check ruler 0 "diff (jump - step) / step -20.00 %" "clears 0"
  # the card's ruler beside it: card.py's half_width at the card's rounds and sd.
  RC=0
  BLOOMERY_LEASE_CARD=$ROOT/docs/cards/sj-jump-vs-step-a6000.card bash -c 'source "$1"; OUT=$2; sj_ruler' _ "$SELF" "$C" > "$C/out" 2>&1 || RC=$?
  st_check ruler-card 0 "the card's ruler: h = 2.145 % at 8 rounds" "the effect clears it"

  if [ "$fails" = 0 ]; then
    echo "sj-vs-clef.sh self-test: $n checks ok"
  else
    echo "sj-vs-clef.sh self-test: $fails of $n checks FAILED"
  fi
  [ "$fails" = 0 ]
}

# Sourced (the self-test runs the runner's functions in a bash of its own), the script defines them and runs nothing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case $MODE in
    selftest) sj_selftest ;;
    *) sj_main ;;
  esac
fi
