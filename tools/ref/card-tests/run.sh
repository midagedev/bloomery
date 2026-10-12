#!/usr/bin/env bash
# The card and lease stub tests: tools/ref/card.py on the fixtures here and on docs/cards/, lease_take
# (tools/ref/lease.sh) and its holder naming, tools/ref/lease-hold.sh, the lease's probe (lease_free),
# the holds and box.sh's guard (lease_guard), tools/box.sh's remote command and its read-only
# refusals (stub ssh), tools/ref/card-precheck.sh, tools/gpu-ab.py's card check, the witness fields
# with a failing nvidia-smi, tools/ref/cards.sh's name→UUID resolution (a fake nvidia-smi on
# PATH: the repository is public and carries no UUID), tools/ref/timing-card.sh's local mode, the t
# table's four readers (tdump.sh), tools/gate-batch.sh's script walker (--classes on a stub tree) and the
# cpu guard (cpu-guard.sh).
#
#   tools/ref/card-tests/run.sh
#
# Runs on the Mac (bash 3.2; there is no util-linux flock there, so bin/flock stands in) and on the
# box, and builds nothing. The lease and gpu-ab tests run against a copy of that code in a fresh
# temporary directory whose docs/cards holds the fixtures under run names (lease mode takes no card
# from anywhere else, and no example), and take a lock file of their own there (BLOOMERY_LEASE_LOCK).
# Where /root/bloomery-cpu.lock exists (the box) lease_take refuses that override, so there the lock
# tests become the check of that refusal; the real lease is the box self-test's:
#   tools/ref/lease-hold.sh --card docs/cards/selftest-lease-hold.card -- true
# One line per test, `ok <name>` or `FAIL <name>: <why>` followed by the output; exit 0 iff none failed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
command -v flock > /dev/null || PATH=$HERE/bin:$PATH
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/card-tests.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
n=0 failed=0

# The copy: the lease code, the t table and gpu-ab.py, the tree's cards, and fixtures as the run cards
# docs/cards/hcpre-ab.card (ab, rounds 5) and docs/cards/exclusive.card. tools/box.sh only has to
# exist and name BLOOMERY_BOX_ENV (gpu-ab.py reads it); nothing runs it.
T=$tmp/tree
mkdir -p "$T/tools/ref" "$T/docs/cards" "$tmp/old/tools" "$tmp/nocard/tools/ref"
cp "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/lease-probe.sh" "$ROOT/tools/ref/lease-hold.sh" "$ROOT/tools/ref/card.py" "$ROOT/tools/ref/tdist.py" \
  "$ROOT/tools/ref/card-precheck.sh" "$T/tools/ref/"
cp "$ROOT/tools/gpu-ab.py" "$T/tools/"
cp "$ROOT"/docs/cards/*.card "$HERE/hcpre-ab.card" "$HERE/exclusive.card" "$T/docs/cards/"
echo 'BLOOMERY_BOX_ENV' > "$T/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/old/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/nocard/tools/box.sh"
cp "$ROOT/tools/ref/card.py" "$ROOT/tools/ref/tdist.py" "$tmp/nocard/tools/ref/"

# The depth-serve runner's stub tests (arms, rotation, refusals, the servers' stop, the tables' math; a stub server, no lease) are one
# long process that mostly waits on its stub servers, so it runs beside every test below and is judged at the end. The runner calls
# lease_take, so gate-batch's walker refuses check-recipes for naming it; a *-tests directory is the harness the walker does not read.
bash "$ROOT/tools/ref/depth-serve.sh" --self-test > "$tmp/depth-serve.out" 2>&1 &
depth_pid=$!

pass() { n=$((n + 1)); echo "ok $1"; }
fail() {
  n=$((n + 1)) failed=$((failed + 1))
  echo "FAIL $1: $2"
  [ -z "${3:-}" ] || sed 's/^/    | /' "$3"
}

# run <name> <rc> <pattern> <command…>: the command's rc against <rc>, its output against grep -E.
run() {
  local name=$1 want=$2 pat=$3 rc
  shift 3
  "$@" > "$tmp/out" 2>&1
  rc=$?
  if [ "$rc" != "$want" ]; then
    fail "$name" "rc $rc, want $want" "$tmp/out"
  elif ! grep -Eq -- "$pat" "$tmp/out"; then
    fail "$name" "no line matches /$pat/" "$tmp/out"
  else
    pass "$name"
  fi
}

# has <name> <pattern>: the last command's output also matches.
has() {
  if grep -Eq -- "$2" "$tmp/out"; then pass "$1"; else fail "$1" "no line matches /$2/" "$tmp/out"; fi
}

# card <fixture> <rc> <pattern> [args…]: card.py check on a fixture of this directory.
card() {
  local f=$1 want=$2 pat=$3
  shift 3
  run "check ${f%.card}${1:+ $*}" "$want" "$pat" python3 "$ROOT/tools/ref/card.py" check "$HERE/$f" "$@"
}

# take <tree> <name> <rc> <pattern> <lock> [NAME=value…]: that tree's lease_take in a fresh shell with
# only the given variables; the lock file is <lock>.
take() {
  local tree=$1 name=$2 want=$3 pat=$4 lock=$5
  shift 5
  run "$name" "$want" "$pat" env -u BLOOMERY_LEASE_CARD -u BLOOMERY_CARD -u ROUNDS -u BLOOMERY_AB_ROUNDS \
    -u ROUND_MINUTES -u BLOOMERY_LEASE_HELD BLOOMERY_LEASE_LOCK="$lock" "$@" \
    bash -c 'source "$1" && lease_take && echo "[test] lease_take returned"' _ "$tree/tools/ref/lease.sh"
}

# run_rel <name> <rc> <pattern> <trigger> <command…>: run (above) with the command's output read here as it comes, so the lock bg_lock
# holds (or a FIFO a command waits on, descriptor 8) is released on the command's own words: the first output line that holds <trigger>
# writes the release, and the end of the output does when none did, so a hold never outlives its waiter. The command runs in the
# background and is waited for by its pid. A trigger that has not come within $rel_cap s (the command printed nothing for that long,
# a waiter that waits for the lock for minutes) releases the hold all the same and is a red by name, never a pass.
rel_cap=50
run_rel() {
  local name=$1 want=$2 pat=$3 trig=$4 rc=0 line seen=0 capped=0 rp rrc t0
  shift 4
  rm -f "$tmp/stream"
  mkfifo "$tmp/stream"
  "$@" > "$tmp/stream" 2>&1 &
  rp=$!
  : > "$tmp/out"
  while :; do
    t0=$SECONDS
    if [ "$seen" = 0 ]; then
      IFS= read -t "$rel_cap" -r line
      rrc=$?
    else
      IFS= read -r line
      rrc=$?
    fi
    # bash 3.2 (the Mac) ends a timed read with 1, as the end of the stream does: a read that took the whole cap timed out.
    if [ "$rrc" -gt 128 ] || { [ "$rrc" != 0 ] && [ "$seen" = 0 ] && [ $((SECONDS - t0)) -ge "$rel_cap" ]; }; then
      capped=1 seen=1
      echo go >&8
      continue
    fi
    [ "$rrc" = 0 ] || break
    printf '%s\n' "$line" >> "$tmp/out"
    case $line in *"$trig"*) if [ "$seen" = 0 ]; then echo go >&8; seen=1; fi ;; esac
  done < "$tmp/stream"
  [ "$seen" = 1 ] || echo go >&8
  wait "$rp" || rc=$?
  if [ "$capped" = 1 ]; then
    fail "$name" "the release trigger '$trig' did not come within $rel_cap s: the hold was released on its cap" "$tmp/out"
  elif [ "$rc" != "$want" ]; then
    fail "$name" "rc $rc, want $want" "$tmp/out"
  elif ! grep -Eq -- "$pat" "$tmp/out"; then
    fail "$name" "no line matches /$pat/" "$tmp/out"
  else
    pass "$name"
  fi
}
# take_rel <tree> <name> <rc> <pattern> <lock> <trigger> [NAME=value…]: take (below) through run_rel.
take_rel() {
  local tree=$1 name=$2 want=$3 pat=$4 lock=$5 trig=$6
  shift 6
  run_rel "$name" "$want" "$pat" "$trig" env -u BLOOMERY_LEASE_CARD -u BLOOMERY_CARD -u ROUNDS -u BLOOMERY_AB_ROUNDS \
    -u ROUND_MINUTES -u BLOOMERY_LEASE_HELD BLOOMERY_LEASE_LOCK="$lock" "$@" \
    bash -c 'source "$1" && lease_take && echo "[test] lease_take returned"' _ "$tree/tools/ref/lease.sh"
}

# hold <tree> <name> <rc> <pattern> <lock> <lease-hold args…>: that tree's lease-hold.sh.
hold() {
  local tree=$1 name=$2 want=$3 pat=$4 lock=$5
  shift 5
  run "$name" "$want" "$pat" env -u BLOOMERY_LEASE_CARD -u BLOOMERY_LEASE_HELD BLOOMERY_LEASE_LOCK="$lock" \
    "$tree/tools/ref/lease-hold.sh" "$@"
}

# gpuab <name> <rc> <pattern> <BLOOMERY_BOX_ENV> <gpu-ab run args…>: the copy's gpu-ab.py, dry run.
gpuab() {
  local name=$1 want=$2 pat=$3 box_env=$4
  shift 4
  run "$name" "$want" "$pat" env BLOOMERY_BOX_ENV="$box_env" python3 "$T/tools/gpu-ab.py" run --dry-run \
    --out "$tmp/gpuab" --recipe time-gpu-generate "$@"
}

absent() {
  if [ -e "$2" ]; then fail "$1" "$2 exists: the lease file was opened before the refusal"; else pass "$1"; fi
}
# lock_state <lock>: the copy's lease_free on <lock> — free, held, or untestable (rc N).
lock_state() {
  local rc=0
  bash -c 'source "$1" && lease_free "$2"' _ "$T/tools/ref/lease-probe.sh" "$1" 2> /dev/null || rc=$?
  case $rc in 0) echo free ;; 1) echo held ;; *) echo "untestable (rc $rc)" ;; esac
}
free() {
  local s
  s=$(lock_state "$2")
  if [ "$s" = free ]; then pass "$1"; else fail "$1" "$2 is $s"; fi
}
held() {
  local s
  s=$(lock_state "$2")
  if [ "$s" = held ]; then pass "$1"; else fail "$1" "$2 is $s"; fi
}
# bg_lock <mode> <lock>: a background process that holds <lock> (-s shared, -x exclusive) from the moment it has it (it says so on a
# FIFO this shell reads, descriptor 7) until bg_end or a trigger line (run_rel) releases it (a line on a second FIFO, descriptor 8; both
# are held open read-write by this shell, so neither side blocks on opening one, and a shell that ends releases the lock by EOF). Its pid
# is in BG. No clock decides how long the lock is held: a case runs its assertions against a lock that is surely held, then ends the hold.
# The 60 s bound is the longest a lock may take to be taken, as a red.
bg_lock() {
  local mode=$1 lock=$2
  exec 7>&- 8>&-
  rm -f "$tmp/rdy" "$tmp/rel"
  mkfifo "$tmp/rdy" "$tmp/rel"
  exec 7<> "$tmp/rdy" 8<> "$tmp/rel"
  if [ "$mode" = -s ]; then
    flock -s "$lock" bash -c 'echo up > "$1"; read -t 120 -r _ < "$2"' _ "$tmp/rdy" "$tmp/rel" &
  else
    flock "$lock" bash -c 'echo up > "$1"; read -t 120 -r _ < "$2"' _ "$tmp/rdy" "$tmp/rel" &
  fi
  BG=$!
  read -t 60 -r _ <&7 && return 0
  echo "bg_lock: $lock was not taken within 60 s" >&2
  echo go >&8
  return 1
}
# bg_end: release what bg_lock holds and wait for it to end.
# The holder's 120 s bound turns a lost release into a red, never a hang.
bg_end() {
  echo go >&8
  wait "$BG" || fail 'bg_lock: the holder' "ended rc $? (no release within its 120 s bound)"
  exec 7>&- 8>&-
}

# fakeproc <dir> <lock file>: a /proc tree for lease_holders (BLOOMERY_LEASE_PROC) with one process,
# pid 4242 (`sleep 60`, card selftest-lease-hold, started 600 s ago), holding <lock file> on
# descriptor 9, and a locks table whose taker, pid 4241, is gone — the runner that died and left its
# child holding the lease through the descriptor it passed down.
fakeproc() {
  local p=$1 lock=$2 key
  key=$(python3 -c 'import os, sys; st = os.stat(sys.argv[1]); print("%02x:%02x:%d" % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))' "$lock")
  mkdir -p "$p/4242/fd" "$p/4242/fdinfo"
  echo '1000.00 3000.00' > "$p/uptime"
  ln -s "$lock" "$p/4242/fd/9"
  ln -s /dev/null "$p/4242/fd/1"
  printf 'pos:\t0\nflags:\t0100001\nlock:\t1: FLOCK  ADVISORY  WRITE 4241 %s 0 EOF\n' "$key" > "$p/4242/fdinfo/9"
  printf 'pos:\t0\nflags:\t01\n' > "$p/4242/fdinfo/1"
  echo sleep > "$p/4242/comm"
  ln -s /bin/sleep "$p/4242/exe"
  ln -s /tmp "$p/4242/cwd"
  printf 'sleep\00060\000' > "$p/4242/cmdline"
  printf 'PATH=/bin\000BLOOMERY_LEASE_CARD=docs/cards/selftest-lease-hold.card\000' > "$p/4242/environ"
  python3 -c 'import os, sys; clk = os.sysconf("SC_CLK_TCK"); print("4242 (sleep) S " + " ".join(["0"] * 18) + " %d 0" % (400 * clk))' > "$p/4242/stat"
  echo "1: FLOCK  ADVISORY  WRITE 4241 $key 0 EOF" > "$p/locks"
}

# card.py on the fixtures: every refusal names its code, every pass its verdict.
card valid-ab.card 0 'ok kind=ab h = 1\.038 % at 4 rounds \(the card; t 2\.447, df 6'
card hcpre-ab.card 0 'ok kind=ab h = 0\.875 % at 5 rounds \(the card; t 2\.306, df 8'
card under-ruler.card 68 'refused CARD_UNDER_RULER \(rc 68\)'
card under-ruler.card 68 'resolves at 5 rounds: h = 0\.875 % < 0\.9 %'
card under-ruler.card 68 'box minutes at 5 rounds: 5 x 2\.5 = 12\.5 min' --round-minutes 2.5
card under-ruler.card 68 'box minutes: unknown'
card under-ruler.card 0 'the runner runs 5 rounds and the card says 3: checked at 5' --rounds 5
card valid-ab.card 68 'the runner runs 2 rounds and the card says 4: checked at 2' --rounds 2
card straddle.card 68 'contains 0, so its nearest edge is 0: no round count resolves it'
card sd-override.card 0 'h = 1\.020 % at 3 rounds .*sd 0\.45 % — a paired ratio SD'
card sd-no-source.card 65 'names no source'
card same-decisions.card 67 'refused CARD_UNDECIDABLE \(rc 67\)'
card no-condition.card 65 'no `condition` \(kind ab requires it\)'
card baseline-no-reason.card 65 'no `reason` \(kind baseline requires it\)'
card profile.card 0 'ok kind=profile'
card profile.card 0 'ok kind=profile' --rounds 3
card profile-rounds.card 65 'kind profile takes no `rounds`'
card exclusive.card 0 'ok kind=exclusive minutes=5\.\.15'
# 2026-10-12: AGENTS.md retired the 30-minute approval rule on 2026-10-10 (a box job's length needs no approval,
# only its predicted wall); the note now says so.
card exclusive-long.card 0 'over 30 minutes is a long box job: it needs no approval, only its predicted wall'
card exclusive-no-minutes.card 65 'no `minutes` \(kind exclusive requires it\)'
card exclusive-predict.card 65 'kind exclusive takes no `predict`'
card unknown-key.card 65 "unknown key 'owner'"
card unknown-kind.card 65 "unknown kind 'benchmark'"
card bad-band.card 65 'is not a band lo\.\.hi'
card reversed-band.card 65 'lo 1\.3 is above hi 0\.9'
card dup-key.card 65 "line 8: 'predict' again \(first on line 6\)"
card ab-unit.card 65 'an ab card predicts its effect in percent'
card continuation.card 0 'ok kind=ab'
card orphan-indent.card 65 'continues no key line above it'
card missing.card 66 'refused CARD_ABSENT'
card noninf-valid.card 0 'ok kind=noninf h_p = 0\.891 % at 6 rounds \(the card; paired, t 2\.571, df 5;.*margin 2 % — the fixture.s flip threshold; lo - h_p = -1\.891 % > -2 %'
card noninf-default.card 71 'refused CARD_UNDER_MARGIN \(rc 71\)'
card noninf-default.card 71 'the band.s lower edge -1 % is at or below -1 %: no round count clears it'
card noninf-rounds.card 71 'at 3 rounds the card would need a margin above 2\.608 %'
card noninf-rounds.card 71 'clears at 14 rounds: lo - h_p = -0\.990 % > -1 %'
card noninf-rounds.card 71 'box minutes at 14 rounds: 14 x 3 = 42 min' --round-minutes 3
card noninf-rounds.card 0 'the runner runs 14 rounds and the card says 3: checked at 14' --rounds 14
card noninf-margin-no-reason.card 65 'names no reason'
# card.py verdict on synthetic ratio lists: the interval is ratio_table's (tools/ref/depth-ds41.sh) to 4 digits.
run 'verdict pass' 0 '^card: verdict: pass: the lower bound -0\.381 % is above -2 %$' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" r1 1.010 r2 1.000 r3 0.995 r4 1.005 r5 1.012 r6 0.998
run 'verdict pass: the ratio line' 0 '^card: mean 1\.0033 ± 0\.0071 \(n=6\): effect \+0\.333 %, lower bound -0\.381 %$' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" r1 1.010 r2 1.000 r3 0.995 r4 1.005 r5 1.012 r6 0.998
run 'verdict fail' 1 '^card: verdict: fail: the lower bound -3\.065 % is at or below -2 %$' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" 0.970 0.975 0.980 0.972 0.978 0.969
run 'verdict fail: the band region' 1 '^card: band: the effect -2\.600 % is below -1\.\.5: decide-below: ' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" 0.970 0.975 0.980 0.972 0.978 0.969
run 'verdict on an ab card' 64 'verdict reads a noninf card; .*valid-ab\.card is kind ab' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/valid-ab.card" 1.01 1.02
run 'verdict on a label with no ratio' 64 "the label 'r2' is followed by no ratio" \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" r1 1.01 r2
run 'verdict on one ratio' 64 'an interval needs 2 rounds or more' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-valid.card" 1.01
run 'verdict on a refused card' 71 'refused CARD_UNDER_MARGIN' \
  python3 "$ROOT/tools/ref/card.py" verdict "$HERE/noninf-default.card" 1.01 1.02
run 'check usage' 64 'usage' python3 "$ROOT/tools/ref/card.py" check
run 'check bad --rounds' 64 '--rounds takes a positive integer' python3 "$ROOT/tools/ref/card.py" check "$HERE/valid-ab.card" --rounds three

# The cards the tree carries: every one passes check; lease mode takes the self-test card and refuses
# the examples, which document the format and predict nothing.
for f in "$ROOT"/docs/cards/*.card; do
  run "docs/cards/${f##*/}" 0 '^card: ok kind=' python3 "$ROOT/tools/ref/card.py" check "$f"
done
run 'lease mode refuses an example card' 66 'docs/cards/example-ab\.card is an example: the format.s documentation, not a prediction' \
  python3 "$ROOT/tools/ref/card.py" lease docs/cards/example-ab.card
run 'lease mode takes the self-test card' 0 '^\[lease\] card: ok kind=exclusive minutes=1 ' \
  python3 "$ROOT/tools/ref/card.py" lease docs/cards/selftest-lease-hold.card

# gpu-ab.py checks the card once before any arm, on this machine: a base tree on an old commit takes
# its lease unchecked. Every arm's BLOOMERY_BOX_ENV is the caller's, then the arm's, then the rounds.
gpuab 'gpu-ab without a card' 66 'refused CARD_ABSENT.*BLOOMERY_LEASE_CARD is empty' '' \
  --rounds 5 --arm base="$tmp/old" --arm new="$T"
gpuab 'gpu-ab checks the card at its rounds' 68 'resolves at 5 rounds' 'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card' \
  --rounds 3 --arm base="$tmp/old" --arm new="$T"
gpuab 'gpu-ab refuses an example card' 66 'is an example' 'BLOOMERY_LEASE_CARD=docs/cards/example-ab.card' \
  --rounds 5 --arm base="$tmp/old" --arm new="$T"
gpuab 'gpu-ab passes the card and plans' 0 '^gpu-ab plan: 3 arms x 5 rounds' \
  'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_CARD_BUDGET=38G' \
  --rounds 5 --arm base="$tmp/old" --arm new="$T" --arm lever="$T:BLOOMERY_X=1"
has 'gpu-ab names the tree that checks nothing' "^gpu-ab: $tmp/old predates the card check"
has 'gpu-ab merges BLOOMERY_BOX_ENV' "BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab\\.card BLOOMERY_CARD_BUDGET=38G BLOOMERY_AB_ROUNDS=5'"
has 'gpu-ab puts the arm levers after the caller' "BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab\\.card BLOOMERY_CARD_BUDGET=38G BLOOMERY_X=1 BLOOMERY_AB_ROUNDS=5'"
gpuab 'gpu-ab refuses an arm tree without the card' 66 "$tmp/nocard checks cards and has no docs/cards/hcpre-ab\\.card" \
  'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card' --rounds 5 --arm base="$tmp/nocard" --arm new="$T"
gpuab 'gpu-ab refuses an arm that sets the rounds' 2 'belongs to the run, not an arm' 'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card' \
  --rounds 5 --arm base="$tmp/old" --arm new="$T:BLOOMERY_AB_ROUNDS=3"
gpuab 'gpu-ab refuses rounds that disagree' 2 'has BLOOMERY_AB_ROUNDS=4 and --rounds is 5' \
  'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_AB_ROUNDS=4' --rounds 5 --arm base="$tmp/old" --arm new="$T"

# lease-hold.sh's own refusals, before any lease.
hold "$T" 'lease-hold without --card' 64 'no --card' "$tmp/h2.lock" -- true
hold "$T" 'lease-hold without a command' 64 'no command after --' "$tmp/h2.lock" --card docs/cards/exclusive.card --
run 'lease-hold with two cards' 64 'name two cards' env BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_LEASE_LOCK="$tmp/h2.lock" \
  "$T/tools/ref/lease-hold.sh" --card docs/cards/exclusive.card -- true

# A GPU timing runner (timing-card.sh sourced: witness_card defined) with TIMING_GPU empty is refused
# before the card or the lease file: its record would say `none`, and tools/gpu-gate.sh would let a
# forced gate onto the card it times. First on the box too.
run 'lease_take refuses a GPU runner with no TIMING_GPU' 64 'refused: a GPU timing runner \(timing-card\.sh sourced\) with TIMING_GPU empty' \
  env -u BLOOMERY_LEASE_HELD BLOOMERY_LEASE_LOCK="$tmp/g0.lock" BLOOMERY_LEASE_CARD=docs/cards/exclusive.card \
  bash -c 'source "$1" && witness_card() { :; } && TIMING_GPU= && lease_take' _ "$T/tools/ref/lease.sh"
absent 'the refused GPU runner opens no lease file' "$tmp/g0.lock"

if [ -e /root/bloomery-cpu.lock ]; then
  # The box: another lock file is refused before anything is opened.
  take "$ROOT" 'lease_take refuses BLOOMERY_LEASE_LOCK on the box' 64 \
    'refused: BLOOMERY_LEASE_LOCK=.* a run under another lock would share the box with a real sitting' "$tmp/b.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/selftest-lease-hold.card
  absent 'the refused override opens no lock file' "$tmp/b.lock"
  hold "$ROOT" 'lease-hold refuses BLOOMERY_LEASE_LOCK on the box' 64 'refused: BLOOMERY_LEASE_LOCK=' "$tmp/b.lock" \
    --card docs/cards/selftest-lease-hold.card -- true
  absent 'lease-hold refused: no lock file opened' "$tmp/b.lock"
else
  # lease_take: a refused card exits with its code before the lease file is opened.
  take "$T" 'lease_take without a card' 66 'refused CARD_ABSENT \(rc 66\): BLOOMERY_LEASE_CARD is empty' "$tmp/l1.lock"
  take "$T" 'lease_take without a card says how to pass one' 66 \
    'no card: a run under the lease needs BLOOMERY_LEASE_CARD=docs/cards/<slug>\.card, passed through BLOOMERY_BOX_ENV' "$tmp/l1.lock"
  absent 'lease_take without a card opens no lease file' "$tmp/l1.lock"
  take "$T" 'lease_take names box.sh BLOOMERY_CARD' 66 "that name is tools/box.sh's card pick" "$tmp/l1.lock" BLOOMERY_CARD=a6000
  take "$T" 'lease_take refuses a card outside docs/cards' 66 'is not a tree-relative docs/cards/<slug>.card path' "$tmp/l1.lock" \
    BLOOMERY_LEASE_CARD=tools/ref/card-tests/valid-ab.card
  take "$T" 'lease_take refuses an example card' 66 'is an example' "$tmp/l1.lock" BLOOMERY_LEASE_CARD=docs/cards/example-profile.card
  take "$T" 'lease_take refuses a missing card' 66 'no such card in' "$tmp/l1.lock" BLOOMERY_LEASE_CARD=docs/cards/no-such.card
  take "$T" 'lease_take checks an ab card at the runner ROUNDS' 68 'h = 1\.360 % at 3 rounds \(the runner' "$tmp/l1.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card ROUNDS=3
  take "$T" 'lease_take falls back to BLOOMERY_AB_ROUNDS' 68 'the runner runs 3 rounds and the card says 5' "$tmp/l1.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_AB_ROUNDS=3
  take "$T" 'lease_take prices a refusal with ROUND_MINUTES' 68 'box minutes at 5 rounds: 5 x 2\.3 = 11\.5 min' "$tmp/l1.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card ROUNDS=3 ROUND_MINUTES=2.3
  take "$T" 'lease_take refuses inside lease-hold' 64 'inside tools/ref/lease-hold\.sh \(pid 4242\)' "$tmp/l1.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/exclusive.card BLOOMERY_LEASE_HELD=4242
  absent 'refused runs open no lease file' "$tmp/l1.lock"
  take "$T" 'lease_take prints the card, then holds' 0 '^\[lease\] card \| kind: ab$' "$tmp/l2.lock" \
    BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card ROUNDS=5
  for pat in '^\[lease\] card: docs/cards/hcpre-ab\.card sha256=[0-9a-f]{64}$' '^\[lease\] card: ok kind=ab h = 0\.875 % at 5 rounds \(the runner' \
    'is not the machine lease' '^\[lease\] held by pid' '^\[test\] lease_take returned$'; do
    has "lease_take ok prints /$pat/" "$pat"
  done
  card_lines=$(grep -c '^\[lease\] card | ' "$tmp/out")
  body_lines=$(wc -l < "$T/docs/cards/hcpre-ab.card" | tr -d ' ')
  if [ "$card_lines" = "$body_lines" ]; then pass "lease_take prints the whole body ($body_lines lines)"; else
    fail 'lease_take prints the whole body' "$card_lines [lease] card | lines for a $body_lines-line card" "$tmp/out"; fi
  free 'the lease ends with the shell that took it' "$tmp/l2.lock"
  # The timing-card record beside the lock (tools/gpu-gate.sh reads it while the lease is held): the
  # runner's pid and its TIMING_GPU, `none` for a runner that times no GPU. lease_gpu_idle reads the
  # timing card's compute apps through nvidia-smi, so the two runs below that name a timing card
  # carry a stub that answers none (rc 0, no output).
  mkdir -p "$tmp/idle"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/idle/nvidia-smi"
  chmod +x "$tmp/idle/nvidia-smi"
  run 'lease_take records a CPU runner as timing no GPU' 0 '^record matches$' env -u BLOOMERY_LEASE_HELD \
    BLOOMERY_LEASE_LOCK="$tmp/g1.lock" BLOOMERY_LEASE_CARD=docs/cards/exclusive.card \
    bash -c 'source "$1" && lease_take > /dev/null && r=$(cat "$BLOOMERY_LEASE_LOCK.card") && echo "$r" &&
      [ "$r" = "pid=$$ timing_gpu=none" ] && echo "record matches"' _ "$T/tools/ref/lease.sh"
  run "lease_take records a GPU runner's timing card" 0 '^record matches$' env -u BLOOMERY_LEASE_HELD \
    BLOOMERY_LEASE_LOCK="$tmp/g2.lock" BLOOMERY_LEASE_CARD=docs/cards/exclusive.card PATH="$tmp/idle:$PATH" \
    bash -c 'source "$1" && witness_card() { :; } && TIMING_GPU=GPU-test && lease_take > /dev/null &&
      r=$(cat "$BLOOMERY_LEASE_LOCK.card") && echo "$r" && [ "$r" = "pid=$$ timing_gpu=GPU-test" ] && echo "record matches"' \
    _ "$T/tools/ref/lease.sh"
  # timing-card.sh's two-card mode: both cards, the A6000 (TIMING_GPU) first; gpu-gate.sh reads it as a doubt.
  run "lease_take records both cards of a two-card run" 0 '^record matches$' env -u BLOOMERY_LEASE_HELD \
    BLOOMERY_LEASE_LOCK="$tmp/g3.lock" BLOOMERY_LEASE_CARD=docs/cards/exclusive.card PATH="$tmp/idle:$PATH" \
    bash -c 'source "$1" && [ "$LEASE_CARDS_RECORD" = 1 ] && witness_card() { :; } && TIMING_GPU=GPU-a && TIMING_GPU2=GPU-b &&
      lease_take > /dev/null && r=$(cat "$BLOOMERY_LEASE_LOCK.card") && echo "$r" && [ "$r" = "pid=$$ timing_gpu=GPU-a,GPU-b" ] &&
      echo "record matches"' _ "$T/tools/ref/lease.sh"
  # tools/box.sh's local mode: the machine lease is bloomery-cpu.lock in BLOOMERY_LOCK_DIR.
  mkdir -p "$tmp/ld-take"
  run 'lease_take under BLOOMERY_LOCK_DIR takes the machine lease' 0 '^\[test\] lease_take returned$' \
    env -u BLOOMERY_LEASE_HELD -u BLOOMERY_LEASE_LOCK -u BLOOMERY_LEASE_TRACE BLOOMERY_LOCK_DIR="$tmp/ld-take" \
    BLOOMERY_LEASE_CARD=docs/cards/exclusive.card bash -c 'source "$1" && lease_take && echo "[test] lease_take returned"' _ "$T/tools/ref/lease.sh"
  if grep -q 'is not the machine lease' "$tmp/out"; then fail 'lease_take under BLOOMERY_LOCK_DIR is admissible' 'it says it is not the machine lease' "$tmp/out"; else
    pass 'lease_take under BLOOMERY_LOCK_DIR is admissible'; fi

  # lease-hold.sh: the card, the lease for the command's life, both witnesses, the command's rc, and a
  # lease that ends with this process whatever the command did.
  L=$tmp/h.lock
  hold "$T" 'lease-hold returns the failing command rc' 1 '^\[lease-hold\] rc=1 held=' "$L" --card docs/cards/exclusive.card -- false
  for pat in '^\[lease\] card \| kind: exclusive$' '^\[lease\] card: ok kind=exclusive minutes=5\.\.15' '^--- witness pre .* epoch [0-9]+ ---$' \
    '^--- witness post .* epoch [0-9]+ ---$' '^\[lease-hold\] command: false$' '^\[lease\] released at '; do
    has "lease-hold prints /$pat/" "$pat"
  done
  free 'lease-hold released the lease after the command failed' "$L"
  hold "$T" 'lease-hold holds the lease while the command runs' 0 '^\[lease-hold\] rc=0 ' "$L" --card docs/cards/exclusive.card -- \
    bash -c 'source "$2" && if lease_free "$1"; then exit 3; fi' _ "$L" "$T/tools/ref/lease.sh"
  hold "$T" 'lease-hold marks its command' 0 '^held-by=[0-9]+$' "$L" --card docs/cards/exclusive.card -- \
    bash -c 'echo "held-by=$BLOOMERY_LEASE_HELD"'
  # A process the command leaves running inherits descriptor 9 and keeps the lease until it exits
  # (heavy work beside the next sitting would contaminate it); the next waiter names it, then takes
  # the lease when it ends. The waiter reads the fake /proc tree (the Mac has none): what is tested
  # here is that a busy lease prints its holder at once; the real /proc is the box's run.
  # The process left behind waits on a FIFO this shell holds open (descriptor 8) until the waiter below has named it: the lease is
  # held when the waiter looks, whatever the load.
  exec 8>&-
  rm -f "$tmp/rel"
  mkfifo "$tmp/rel"
  exec 8<> "$tmp/rel"
  hold "$T" 'lease-hold command leaves a process behind' 0 '^\[lease-hold\] rc=0 ' "$L" --card docs/cards/exclusive.card -- \
    bash -c '(read -r _ < "$1") > /dev/null 2>&1 & exit 0' _ "$tmp/rel"
  has 'lease-hold says the lease outlives it' '^\[lease\] this process let go at .*, but the lease is still held'
  held 'the process left behind keeps the lease' "$L"
  fakeproc "$tmp/proc" "$L"
  # lease_holders prints once it has scanned the /proc tree, so its first line (`[lease]   `) releases the process left behind.
  take_rel "$T" 'a waiter names the holder at once' 0 "^\\[lease\\] $L is held; waiting up to 30 min\\. Its holder:\$" "$L" '[lease]   ' \
    BLOOMERY_LEASE_CARD=docs/cards/exclusive.card BLOOMERY_LEASE_PROC="$tmp/proc"
  exec 8>&-
  for pat in '^\[lease\]   pid 4242 holds it: comm=sleep exe=/bin/sleep cwd=/tmp elapsed=0h10m00s card=docs/cards/selftest-lease-hold\.card args=\[sleep 60\]$' \
    '/locks: FLOCK WRITE taken by pid 4241 \(not running — a lock outlives the process that took it .*; the holders are the processes above\)$' \
    '^\[lease\] held by pid' '^\[test\] lease_take returned$'; do
    has "the waiter prints /$pat/" "$pat"
  done
  if grep -q 'pid 4242 .*fd/1\|has it open' "$tmp/out"; then fail 'the waiter lists only the descriptor on the lock' 'a line names another descriptor' "$tmp/out"; else
    pass 'the waiter lists only the descriptor on the lock'; fi
  free 'the lease is free once the process left behind exits' "$L"
  run 'lease_holders on a lock nobody holds says so' 0 '^\[lease\]   no holder found: no process has .* locked in ' \
    env BLOOMERY_LEASE_PROC="$tmp/proc-empty" bash -c 'mkdir -p "$2" && : > "$3" && source "$1" && lease_holders "$3"' _ "$T/tools/ref/lease.sh" "$tmp/proc-empty" "$tmp/free.lock"
  run 'lease_holders without a /proc tree says so' 0 'cannot be named: .*/no-proc cannot be listed' \
    env BLOOMERY_LEASE_PROC="$tmp/no-proc" bash -c 'source "$1" && lease_holders "$2"' _ "$T/tools/ref/lease.sh" "$L"
  # A probe (lease_free's shared lock) in at the instant lease_take tries its exclusive `flock -n 9`
  # fails that try: the runner waits (`flock -w 60 9`) and takes the lease once the probe lets go.
  if bg_lock -s "$tmp/l3.lock"; then
    take_rel "$T" 'lease_take meeting a shared probe waits for it' 0 "^\\[lease\\] $tmp/l3\\.lock is held; waiting up to 30 min\\. Its holder:\$" \
      "$tmp/l3.lock" '[lease]   ' BLOOMERY_LEASE_CARD=docs/cards/exclusive.card BLOOMERY_LEASE_PROC="$tmp/proc-empty"
    has 'lease_take takes the lease once the probe lets go' '^\[lease\] held by pid'
    bg_end
  else
    fail 'lease_take meeting a shared probe' 'the background shared lock was not taken'
  fi
  hold "$T" 'lease-hold refuses a card outside docs/cards' 66 'refused CARD_ABSENT' "$tmp/h2.lock" --card tools/ref/card-tests/valid-ab.card -- true
  absent 'lease-hold refused: no lease file opened' "$tmp/h2.lock"
  hold "$ROOT" 'this tree: lease-hold with the self-test card' 0 '^\[lease\] card: ok kind=exclusive minutes=1 ' "$tmp/h3.lock" \
    --card docs/cards/selftest-lease-hold.card -- true
  free 'this tree: the self-test lease is released' "$tmp/h3.lock"
fi

# rc_is <name> <rc> <command…>: the command's exit code only (its output may be empty).
rc_is() {
  local name=$1 want=$2 rc=0
  shift 2
  "$@" > "$tmp/out" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then pass "$name"; else fail "$name" "rc $rc, want $want" "$tmp/out"; fi
}
# silent <name>: the last command printed nothing.
silent() {
  if [ -s "$tmp/out" ]; then fail "$1" 'it printed' "$tmp/out"; else pass "$1"; fi
}

# lease_free, the one probe of the lease: a shared lock, so another probe (shared) never reads as held
# and only an exclusive holder does; a lock it cannot open is untestable (rc 2) and is not free in any
# `if`/`||` form. The old exclusive probe under the same shared holder reads it as held: the false read.
# These take lock files of their own (not the lease), so they run on the box too.
P=$tmp/probe.lock
: > "$P"
probe() { bash -c 'source "$1" && lease_free "$2"' _ "$T/tools/ref/lease-probe.sh" "$1"; }
rc_is 'lease_free: a free lock reads free (0)' 0 probe "$P"
if bg_lock -s "$P"; then
  rc_is 'lease_free: a shared holder (another probe) does not read as held (0)' 0 probe "$P"
  rc_is 'the old exclusive probe reads that shared holder as held (1, the false read)' 1 flock -n "$P" true
  bg_end
else
  fail 'lease_free: a shared holder' 'the background shared lock was not taken'
fi
if bg_lock -x "$P"; then
  rc_is 'lease_free: an exclusive holder (the lease) reads held (1)' 1 probe "$P"
  bg_end
else
  fail 'lease_free: an exclusive holder' 'the background exclusive lock was not taken'
fi
run 'lease_free: a lock it cannot open is untestable (2), named' 2 'no-such-dir/x\.lock cannot be tested \(flock rc 66\): read as not free' \
  probe "$tmp/no-such-dir/x.lock"
run 'if lease_free reads an untestable lock as not free' 0 '^not free$' \
  bash -c 'source "$1" && if lease_free "$2" 2> /dev/null; then echo free; else echo "not free"; fi' _ "$T/tools/ref/lease-probe.sh" "$tmp/no-such-dir/x.lock"

# The holds and lease_guard (what tools/box.sh runs before a command), from lease-probe.sh alone — the
# file box.sh sources — against a lock and a hold directory of their own; BLOOMERY_LEASE_POLL=1 so a
# wait takes seconds.
H=$tmp/holds
mkdir -p "$H"
guardenv() {
  env BLOOMERY_LEASE_LOCK="$P" BLOOMERY_LEASE_HOLDS="$H/bloomery-*-hold" BLOOMERY_LEASE_POLL=1 \
    BLOOMERY_LEASE_PROC="$tmp/proc-none" bash -c 'source "$1" && shift && "$@"' _ "$T/tools/ref/lease-probe.sh" "$@"
}
guard() {
  local name=$1 want=$2 pat=$3
  shift 3
  run "$name" "$want" "$pat" guardenv "$@"
}
rc_is 'no hold up: lease_holds_up is 0' 0 guardenv lease_holds_up
silent 'no hold up: lease_holds_up prints nothing'
touch "$H/bloomery-03-hold"
guard 'a hold up: lease_holds_up names it, its owner, age and start' 0 "^$H/bloomery-03-hold owner=03 up=0h00m0[0-9]s since=[0-9]+\$" lease_holds_up
rc_is "the owner's own hold is passed" 0 guardenv lease_holds_up 03
silent "the owner's own hold is passed: nothing printed"
touch "$H/bloomery-aa-hold"
guard "another owner's hold is not passed" 0 "^$H/bloomery-aa-hold owner=aa " lease_holds_up 03
if grep -q 'bloomery-03-hold' "$tmp/out"; then fail "the owner's own hold is not listed to it" 'the 03 hold is listed to owner 03' "$tmp/out"; else
  pass "the owner's own hold is not listed to it"; fi
guard 'lease_guard 0 with a hold up: 75 at once' 75 'BLOOMERY_BOX_WAIT=0 does not wait: the command did not run \(rc 75\)' lease_guard 0 03
has 'lease_guard names the hold, its owner and how the owner passes' "hold $H/bloomery-aa-hold owner=aa up=.* its owner passes with BLOOMERY_HOLD_OWNER"
rm -f "$H/bloomery-aa-hold"
rc_is "lease_guard 0 with only the caller's own hold up: 0" 0 guardenv lease_guard 0 03
silent "lease_guard 0 with only the caller's own hold up: silent"
guard "lease_guard 0 with another owner's hold up: 75" 75 'did not run \(rc 75\)' lease_guard 0 aa
rm -f "$H/bloomery-03-hold"
rc_is 'lease_guard 0 on a quiet box: 0 at once' 0 guardenv lease_guard 0
silent 'lease_guard 0 on a quiet box: silent'
if bg_lock -x "$P"; then
  guard 'lease_guard 0 with the lease held: 75, naming the lease' 75 'the timing lease .*probe\.lock is held \(a sitting runs\); its holders:' lease_guard 0
  has 'lease_guard asks lease_holders for the holder' '^\[lease\]   the holder of .* cannot be named'
  bg_end
else
  fail 'lease_guard with the lease held' 'the background exclusive lock was not taken'
fi
if bg_lock -x "$P"; then
  # The lease is released on the guard's own "waits" line, so it is held when the guard first looks and free for its two quiet polls.
  run_rel 'lease_guard waits for the lease, then starts on two quiet polls' 0 'quiet on two polls in a row after [0-9]+ s: the command starts' \
    'the box is busy; the command waits up to 20 s' guardenv lease_guard 20
  has 'lease_guard says it waits, with its bound and poll' 'the box is busy; the command waits up to 20 s \(BLOOMERY_BOX_WAIT\), polling every 1 s:'
  bg_end
else
  fail 'lease_guard waits for the lease' 'the background exclusive lock was not taken'
fi
touch "$H/bloomery-aa-hold"
guard 'lease_guard gives up at its bound: 75, naming what is up' 75 'still busy after [0-9]+ s \(the bound, BLOOMERY_BOX_WAIT=2\): the command did not run \(rc 75\)' lease_guard 2
rm -f "$H/bloomery-aa-hold"
# Two holds up, each owner running its own sitting: the hold that went up later gives way at once.
touch -t 202601010000 "$H/bloomery-03-hold"
touch -t 202601010100 "$H/bloomery-aa-hold"
guard 'two holds up: the later owner gives way at once (75)' 75 'hold .*bloomery-03-hold \(owner 03\) went up before .*bloomery-aa-hold: two sittings would wait on each other, so this one gives way' lease_guard 20 aa
guard 'two holds up: the earlier owner waits as for any hold' 75 'BLOOMERY_BOX_WAIT=0 does not wait' lease_guard 0 03
if grep -q 'gives way' "$tmp/out"; then fail 'the earlier owner does not give way' 'it gave way' "$tmp/out"; else pass 'the earlier owner does not give way'; fi
touch -t 202601010000 "$H/bloomery-aa-hold"
guard 'two holds up in the same second: the owner that sorts later gives way' 75 'bloomery-03-hold \(owner 03\) went up before' lease_guard 20 aa
rm -f "$H/bloomery-03-hold" "$H/bloomery-aa-hold"
# The same two holds under a lock directory with a space in its path (tools/box.sh's local mode passes it whole).
D="$tmp/lock dir"
mkdir -p "$D"
touch -t 202601010000 "$D/bloomery-03-hold"
touch -t 202601010100 "$D/bloomery-aa-hold"
run 'two holds up under a lock directory with a space: the later owner gives way at once (75)' 75 "hold $D/bloomery-03-hold \\(owner 03\\) went up before $D/bloomery-aa-hold" \
  env -u BLOOMERY_LEASE_HOLDS BLOOMERY_LOCK_DIR="$D" BLOOMERY_LEASE_LOCK="$P" BLOOMERY_LEASE_POLL=1 BLOOMERY_LEASE_PROC="$tmp/proc-none" \
  bash -c 'source "$1" && lease_guard 2 aa' _ "$T/tools/ref/lease-probe.sh"
rm -f "$D/bloomery-03-hold" "$D/bloomery-aa-hold"
run 'lease_guard on an untestable lease: 70 at once' 70 'the lease cannot be tested \(above\): the command did not run \(rc 70\)' \
  env BLOOMERY_LEASE_LOCK="$tmp/no-such-dir/x.lock" BLOOMERY_LEASE_HOLDS="$H/bloomery-*-hold" bash -c 'source "$1" && lease_guard 0' _ "$T/tools/ref/lease-probe.sh"
guard 'lease_guard refuses a bound that is not seconds' 64 'the bound is whole seconds' lease_guard 1m
guard 'lease_guard refuses an owner that is not a word' 64 'an owner is letters, digits and _' lease_guard 0 0/3
run 'lease_guard refuses a hold pattern without one * for the owner' 64 'is not <prefix>\*<suffix> with one' \
  env BLOOMERY_LEASE_LOCK="$P" BLOOMERY_LEASE_HOLDS="$H/hold" bash -c 'source "$1" && lease_guard 0' _ "$T/tools/ref/lease-probe.sh"
# The lease and the holds default to /root, and BLOOMERY_LOCK_DIR moves both (tools/box.sh's local mode).
lockdir() { env -u BLOOMERY_LEASE_LOCK -u BLOOMERY_LEASE_HOLDS "$@" bash -c 'source "$1" && echo "$LEASE_LOCK $LEASE_HOLDS"' _ "$T/tools/ref/lease-probe.sh"; }
run 'the lease and the holds are under /root by default' 0 '^/root/bloomery-cpu\.lock /root/bloomery-\*-hold$' lockdir -u BLOOMERY_LOCK_DIR
run 'BLOOMERY_LOCK_DIR moves the lease and the holds' 0 "^$tmp/ld/bloomery-cpu\\.lock $tmp/ld/bloomery-\\*-hold\$" lockdir BLOOMERY_LOCK_DIR="$tmp/ld"

# tools/box.sh builds the guard into the remote command, before the environment and the command; the
# read-only opt-in leaves it out and refuses a build. ssh and rsync are stubs that record their
# arguments and reach nothing; every refusal happens before the first ssh. The remote command spans
# two lines (box.sh's card pick ends in a newline), so the guard is on the first and the command on
# the last.
B=$tmp/boxtree
mkdir -p "$B/tools" "$tmp/stub"
cp "$ROOT/tools/box.sh" "$B/tools/"
cp "$ROOT/Cargo.toml" "$B/"
# The ssh stub answers the queue's one probe (flock -n -x on the dir lock) with $STUB_PROBE_RC, 0 where unset; the rsync stub
# records its arguments and the .sync-id it was handed, and ends with $STUB_RSYNC_RC, 0 where unset.
printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\ncase "$*" in *"flock -n -x"*) exit ${STUB_PROBE_RC:-0} ;; esac\n' "$tmp/ssh.log" > "$tmp/stub/ssh"
printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\nfor a; do case $a in */.sync-id) cp "$a" "%s"; echo "$a" > "%s" ;; esac; done\nexit ${STUB_RSYNC_RC:-0}\n' \
  "$tmp/rsync.log" "$tmp/rsync.sync-id" "$tmp/rsync.sync-path" > "$tmp/stub/rsync"
# The queue client stub: the words before the job's text, one a line, and the text itself (the last word) byte for byte.
cat > "$B/tools/boxq.py" << 'PYSTUB'
import os, sys
with open(os.environ["STUB_BOXQ_ARGS"], "w") as f:
    f.write("\n".join(sys.argv[1:-1]) + "\n")
with open(os.environ["STUB_BOXQ_JOB"], "w") as f:
    f.write(sys.argv[-1])
sys.exit(int(os.environ.get("STUB_BOXQ_RC", "0")))
PYSTUB
chmod +x "$tmp/stub/ssh" "$tmp/stub/rsync"
# The caller's own box.sh variables are dropped (a batch run with BLOOMERY_HOLD_OWNER set runs these
# tests too), and HOME is a fresh one, so a Mac-side hold under ~/.cache/bloomery/box-hold does not refuse
# them; a test that wants one sets it after `box`.
box() {
  : > "$tmp/ssh.log"
  rm -f "$tmp/rsync.log" "$tmp/rsync.sync-id" "$tmp/rsync.sync-path" "$tmp/boxq.args" "$tmp/boxq.job"
  env -u BLOOMERY_HOLD_OWNER -u BLOOMERY_BOX_WAIT -u BLOOMERY_BOX_READONLY -u BLOOMERY_BOX_ENV -u BLOOMERY_CARD \
    -u BLOOMERY_MODEL -u BLOOMERY_REF_MODEL -u BLOOMERY_V41_MODEL -u BLOOMERY_DATA -u BLOOMERY_ARCH \
    -u BLOOMERY_BOX_QUEUE -u BLOOMERY_Q_CLASS -u BLOOMERY_Q_CARD -u BLOOMERY_Q_PRED_S -u STUB_RSYNC_RC -u STUB_PROBE_RC -u STUB_BOXQ_RC \
    -u BLOOMERY_BOX_LOCAL -u BLOOMERY_ENV_FILE -u BLOOMERY_LOCK_DIR \
    STUB_BOXQ_ARGS="$tmp/boxq.args" STUB_BOXQ_JOB="$tmp/boxq.job" \
    HOME="$tmp/home" PATH="$tmp/stub:$PATH" BLOOMERY_BOX=stub BLOOMERY_REMOTE='~/repo/x' "$@"
}
sshlog() {
  if grep -Eq -- "$2" "$tmp/ssh.log"; then pass "$1"; else fail "$1" "no ssh line matches /$2/" "$tmp/ssh.log"; fi
}
rc_is 'box.sh: a command reaches the stub ssh' 0 box "$B/tools/box.sh" 'cargo check'
sshlog 'box.sh: the guard comes first, before the environment' '^stub \( cd ~/repo/x && \. tools/ref/lease-probe\.sh && lease_guard 1800  \) && source ~/bloomery-env\.sh && \{ :$'
sshlog 'box.sh: the command comes last' '^\} && cd ~/repo/x && .*cargo check$'
# A cargo oxide command also checks that the cargo-oxide on the box's PATH is the pinned rev (an older one ignores the
# [target.<triple>] tables without a word); any other command does not. The check itself runs here against a stub
# `cargo install --list`: the pin passes, another rev and a path install stop with rc 70 naming the pin.
PIN=$(sed -n '/^\[patch\."https:\/\/github.com\/NVIDIA\/cuda-rust.git"\]/,/^\[workspace/s/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$B/Cargo.toml")
rc_is 'box.sh: a cargo oxide command reaches the stub ssh' 0 box "$B/tools/box.sh" 'cargo oxide build'
sshlog 'box.sh: a cargo oxide command checks the installed cargo-oxide against the pin' "cargo install --list .*not the pin $PIN"
rc_is 'box.sh: a cargo oxide command with BLOOMERY_BOX_ENV runs' 0 box env BLOOMERY_BOX_ENV=CARGO_HOME=/opt/c "$B/tools/box.sh" 'cargo oxide build'
sshlog 'box.sh: the cargo-oxide check runs after BLOOMERY_BOX_ENV' 'export CARGO_HOME=/opt/c && .*\{ __c=\$\(cargo install --list'
for cmd in 'bash tools/gate.sh --oxide -p x --release --lib' 'bash tools/release/build.sh 0.1.0 abcdef0' \
  'bash tools/release/build.sh --check /tmp/out && bash tools/release/build.sh 0.1.0 abcdef0'; do
  rc_is "box.sh: '$cmd' reaches the stub ssh" 0 box "$B/tools/box.sh" "$cmd"
  sshlog "box.sh: '$cmd' checks the installed cargo-oxide too" "cargo install --list .*not the pin $PIN"
done
for cmd in 'cargo check' 'bash tools/release/build.sh --check /tmp/out'; do
  rc_is "box.sh: '$cmd' reaches the stub ssh" 0 box "$B/tools/box.sh" "$cmd"
  if grep -q 'cargo install --list' "$tmp/ssh.log"; then fail "box.sh: '$cmd' does not check cargo-oxide" 'the check is in it' "$tmp/ssh.log"; else pass "box.sh: '$cmd' does not check cargo-oxide"; fi
done
OXCLI=$(sed -n "s/^OXCLI='\(.*\)'\$/\1/p" "$B/tools/box.sh")
OXCLI=${OXCLI//OXREV/$PIN}
mkdir -p "$tmp/oxbin"
for got in "${PIN:0:8}" 4da6c138 path; do
  if [ "$got" = path ]; then
    printf '#!/bin/sh\necho "cargo-oxide v0.2.1 (/src/cuda-oxide/crates/cargo-oxide):"\n' > "$tmp/oxbin/cargo"
  else
    printf '#!/bin/sh\necho "cargo-oxide v0.2.1 (https://github.com/midagedev/cuda-oxide.git?rev=x#%s):"\n' "$got" > "$tmp/oxbin/cargo"
  fi
  chmod +x "$tmp/oxbin/cargo"
  if [ "$got" = "${PIN:0:8}" ]; then
    run "box.sh: the pinned cargo-oxide passes the check" 0 '^ran$' env PATH="$tmp/oxbin:/usr/bin:/bin" bash -c "${OXCLI} echo ran"
  else
    run "box.sh: a cargo-oxide from $got is refused by name" 70 "installed cargo-oxide is from rev .*not the pin $PIN" \
      env PATH="$tmp/oxbin:/usr/bin:/bin" bash -c "${OXCLI} echo ran"
  fi
done
rc_is 'box.sh: BLOOMERY_BOX_WAIT and BLOOMERY_HOLD_OWNER reach the guard' 0 box env BLOOMERY_BOX_WAIT=0 BLOOMERY_HOLD_OWNER=03 "$B/tools/box.sh" true
sshlog 'box.sh: lease_guard 0 03' '^stub \( cd ~/repo/x && \. tools/ref/lease-probe\.sh && lease_guard 0 03 \) && source '
rc_is 'box.sh: a read-only command runs' 0 box env BLOOMERY_BOX_READONLY=1 "$B/tools/box.sh" 'ps -e | wc -l'
sshlog 'box.sh: the read-only command has no guard' '^stub source ~/bloomery-env\.sh && \{ :$'
sshlog 'box.sh: the read-only command is the command' '^\} && cd ~/repo/x && .*ps -e \| wc -l$'
for word in 'cargo --version' 'just check' 'make -j' 'cmake ..' 'ninja -C build' './target/release/generate_ds41 --time'; do
  w=${word%% *}
  [ "${w#./target/}" = "$w" ] || w=target/
  run "box.sh: BLOOMERY_BOX_READONLY=1 refuses '$word' by name" 64 "names '$w'" box env BLOOMERY_BOX_READONLY=1 "$B/tools/box.sh" "$word"
  if [ -s "$tmp/ssh.log" ]; then fail "box.sh: the refusal of '$word' reached ssh" 'ssh ran' "$tmp/ssh.log"; else pass "box.sh: the refusal of '$word' reached no ssh"; fi
done
run 'box.sh: BLOOMERY_BOX_READONLY takes 1 or 0' 64 'BLOOMERY_BOX_READONLY is 1' box env BLOOMERY_BOX_READONLY=yes "$B/tools/box.sh" ls
run 'box.sh: BLOOMERY_BOX_WAIT is seconds' 64 'BLOOMERY_BOX_WAIT is whole seconds' box env BLOOMERY_BOX_WAIT=30m "$B/tools/box.sh" ls
run 'box.sh: BLOOMERY_HOLD_OWNER is a word' 64 'BLOOMERY_HOLD_OWNER is the <owner>' box env BLOOMERY_HOLD_OWNER=/root/bloomery-03-hold "$B/tools/box.sh" ls
# BLOOMERY_BOX_ENV splits on spaces: a value with a space is refused by name, never exported in halves.
rc_is 'box.sh: BLOOMERY_BOX_ENV entries reach the command' 0 box env BLOOMERY_BOX_ENV='A=1 B_2=x' "$B/tools/box.sh" true
sshlog 'box.sh: each BLOOMERY_BOX_ENV entry is one export' 'export A=1 && export B_2=x && .*true$'
run 'box.sh: a BLOOMERY_BOX_ENV entry with a quote is refused' 64 "entry 'F=\"-ngl' holds a quote: BLOOMERY_BOX_ENV splits on spaces" \
  box env BLOOMERY_BOX_ENV='F="-ngl 999" X=1' "$B/tools/box.sh" true
run 'box.sh: the second half of a spaced value is refused' 64 "got '16' \\(after 'F=-t': the rest of a value with a space" \
  box env BLOOMERY_BOX_ENV='F=-t 16' "$B/tools/box.sh" true
run 'box.sh: a lower-case name after an entry is refused' 64 "'x' in BLOOMERY_BOX_ENV is not an upper-case variable name \(after 'F=-t'" \
  box env BLOOMERY_BOX_ENV='F=-t x=1' "$B/tools/box.sh" true
if grep -q 'true$' "$tmp/ssh.log"; then fail 'box.sh: a BLOOMERY_BOX_ENV refusal ran the command' 'the command reached ssh' "$tmp/ssh.log"; else pass 'box.sh: a BLOOMERY_BOX_ENV refusal runs no command'; fi
rc_is 'box.sh: a command with the default card runs' 0 box "$B/tools/box.sh" true
sshlog 'box.sh: the default card reaches the box as BLOOMERY_BOX_CARD=3090' ' BLOOMERY_BOX_CARD=3090 && .*true$'
rc_is 'box.sh: a command with BLOOMERY_CARD=both runs' 0 box env BLOOMERY_CARD=both "$B/tools/box.sh" true
sshlog 'box.sh: BLOOMERY_CARD=both reaches the box as BLOOMERY_BOX_CARD=both' ' BLOOMERY_BOX_CARD=both && .*true$'
# The arch (tools/arch.sh): unset adds nothing to the remote command and reads no tools/arch.sh (this stub tree has none yet); set,
# it is exported ahead of the BLOOMERY_BOX_ENV entries; a value that is no arch, or one BLOOMERY_BOX_ENV contradicts, is 64 before
# anything reaches ssh.
rc_is 'box.sh: BLOOMERY_ARCH unset runs without tools/arch.sh' 0 box "$B/tools/box.sh" true
if grep -q 'BLOOMERY_ARCH' "$tmp/ssh.log"; then fail 'box.sh: BLOOMERY_ARCH unset adds nothing to the remote command' 'the command names it' "$tmp/ssh.log"; else pass 'box.sh: BLOOMERY_ARCH unset adds nothing to the remote command'; fi
cp "$ROOT/tools/arch.sh" "$B/tools/"
rc_is 'box.sh: BLOOMERY_ARCH=sm_121 runs' 0 box env BLOOMERY_ARCH=sm_121 "$B/tools/box.sh" true
sshlog 'box.sh: BLOOMERY_ARCH reaches the command as one export' ' export BLOOMERY_ARCH=sm_121 && .*true$'
rc_is 'box.sh: the same BLOOMERY_ARCH in BLOOMERY_BOX_ENV runs' 0 box env BLOOMERY_ARCH=sm_121 BLOOMERY_BOX_ENV=BLOOMERY_ARCH=sm_121 "$B/tools/box.sh" true
for bad in sm121 '' SM_86 sm_8; do
  run "box.sh: BLOOMERY_ARCH='$bad' is refused" 64 "BLOOMERY_ARCH is a CUDA arch.*got '$bad'" box env BLOOMERY_ARCH="$bad" "$B/tools/box.sh" true
  if [ -s "$tmp/ssh.log" ] || [ -e "$tmp/rsync.log" ]; then fail "box.sh: the refused BLOOMERY_ARCH='$bad' reached ssh or rsync" 'something ran' "$tmp/ssh.log"; else pass "box.sh: the refused BLOOMERY_ARCH='$bad' reached no ssh or rsync"; fi
done
run 'box.sh: BLOOMERY_ARCH and a different BLOOMERY_BOX_ENV entry are refused' 64 "BLOOMERY_ARCH is 'sm_86' in BLOOMERY_BOX_ENV and 'sm_121' in the environment \(unset: sm_86\): name it in the environment" \
  box env BLOOMERY_ARCH=sm_121 BLOOMERY_BOX_ENV=BLOOMERY_ARCH=sm_86 "$B/tools/box.sh" true
if [ -s "$tmp/ssh.log" ] || [ -e "$tmp/rsync.log" ]; then fail 'box.sh: the contradicting BLOOMERY_ARCH reached ssh or rsync' 'something ran' "$tmp/ssh.log"; else pass 'box.sh: the contradicting BLOOMERY_ARCH reached no ssh or rsync'; fi
# An entry with the environment unset is measured against the default: the recipes expanded --arch sm_86 on this side.
run 'box.sh: a BLOOMERY_BOX_ENV arch entry that is not the default, with BLOOMERY_ARCH unset, is refused' 64 "BLOOMERY_ARCH is 'sm_121' in BLOOMERY_BOX_ENV and 'sm_86' in the environment \(unset: sm_86\)" \
  box env BLOOMERY_BOX_ENV=BLOOMERY_ARCH=sm_121 "$B/tools/box.sh" true
if [ -s "$tmp/ssh.log" ] || [ -e "$tmp/rsync.log" ]; then fail 'box.sh: the unset-environment BLOOMERY_ARCH entry reached ssh or rsync' 'something ran' "$tmp/ssh.log"; else pass 'box.sh: the unset-environment BLOOMERY_ARCH entry reached no ssh or rsync'; fi
rc_is 'box.sh: a BLOOMERY_BOX_ENV arch entry equal to the default runs' 0 box env BLOOMERY_BOX_ENV=BLOOMERY_ARCH=sm_86 "$B/tools/box.sh" true
sshlog 'box.sh: the entry is the only export of BLOOMERY_ARCH' 'export BLOOMERY_ARCH=sm_86 && .*true$'
if [ "$(grep -o 'export BLOOMERY_ARCH=' "$tmp/ssh.log" | wc -l | tr -d ' ')" = 1 ]; then pass 'box.sh: the entry alone exports BLOOMERY_ARCH once'; else fail 'box.sh: the entry alone exports BLOOMERY_ARCH once' 'not once' "$tmp/ssh.log"; fi
rm -f "$B/tools/arch.sh"

# The box queue (BLOOMERY_BOX_QUEUE=1): the call syncs behind the remote directory's lock and submits the command as a job
# of tools/boxq.py (the stub above: it records its words and the job's text, and ends with $STUB_BOXQ_RC). Where the stub
# tree cannot show a behaviour (the job's own text on the box), the recorded text itself runs here, under stub tools.
Q=$tmp/queue
mkdir -p "$Q"
inlog() {
  if grep -Eq -- "$3" "$2"; then pass "$1"; else fail "$1" "no line of ${2##*/} matches /$3/" "$2"; fi
}
notinlog() {
  if grep -Eq -- "$3" "$2"; then fail "$1" "a line of ${2##*/} matches /$3/" "$2"; else pass "$1"; fi
}
nofile() {
  if [ -e "$2" ]; then fail "$1" "${2##*/} exists" "$2"; else pass "$1"; fi
}
# The unset path: no queue words anywhere, and BLOOMERY_BOX_QUEUE=0 is the same bytes.
rc_is 'box.sh: a direct call' 0 box "$B/tools/box.sh" 'cargo check'
cp "$tmp/ssh.log" "$Q/direct.ssh"; cp "$tmp/rsync.log" "$Q/direct.rsync"
nofile 'box.sh: a direct call runs no queue client' "$tmp/boxq.args"
notinlog 'box.sh: a direct call has no dir lock in its rsync' "$tmp/rsync.log" 'rsync-path|\.sync-id|bloomery-queue'
rc_is 'box.sh: BLOOMERY_BOX_QUEUE=0 is the direct call' 0 box env BLOOMERY_BOX_QUEUE=0 "$B/tools/box.sh" 'cargo check'
if cmp -s "$Q/direct.ssh" "$tmp/ssh.log" && cmp -s "$Q/direct.rsync" "$tmp/rsync.log"; then pass 'box.sh: BLOOMERY_BOX_QUEUE=0 sends the same ssh and rsync arguments as unset'; else fail 'box.sh: BLOOMERY_BOX_QUEUE=0 sends the same ssh and rsync arguments as unset' 'they differ' "$tmp/ssh.log"; fi
box env BLOOMERY_BOX_LOCAL=0 "$B/tools/box.sh" 'cargo check' > "$tmp/out" 2>&1
if cmp -s "$Q/direct.ssh" "$tmp/ssh.log" && cmp -s "$Q/direct.rsync" "$tmp/rsync.log"; then pass 'box.sh: BLOOMERY_BOX_LOCAL=0 sends the same ssh and rsync arguments as unset'; else fail 'box.sh: BLOOMERY_BOX_LOCAL=0 sends the same ssh and rsync arguments as unset' 'they differ' "$tmp/ssh.log"; fi
run 'box.sh: BLOOMERY_BOX_QUEUE takes 1 or 0' 64 'BLOOMERY_BOX_QUEUE is 1' box env BLOOMERY_BOX_QUEUE=yes "$B/tools/box.sh" ls
nofile 'box.sh: a queue refusal syncs nothing' "$tmp/rsync.log"

# A queued call: the lock behind rsync, the id along, the words of the job, no guard, the queue line before the submit.
run 'box.sh: a queued call submits one job' 0 'box.sh: queue round card=3090 predicted 90 s remote=~/repo/x$' \
  box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_PRED_S=90 "$B/tools/box.sh" 'echo hi'
inlog 'box.sh: the sync runs behind the remote directory'"'"'s lock' "$tmp/rsync.log" \
  ' --rsync-path=mkdir -p /root/bloomery-queue && flock -n -x /root/bloomery-queue/dir-x\.lock rsync '
inlog 'box.sh: the id goes with the tree as .sync-id, after the tree' "$tmp/rsync.log" ' '"$B"'/ [^ ]*/\.sync-id stub:~/repo/x/$'
id=$(cat "$tmp/rsync.sync-id" 2> /dev/null)
case $id in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) pass 'box.sh: the id is 16 hex digits' ;;
  *) fail 'box.sh: the id is 16 hex digits' "got '$id'" ;;
esac
if [ -e "$(cat "$tmp/rsync.sync-path" 2> /dev/null || echo "$tmp/none")" ]; then fail 'box.sh: the id'"'"'s temp dir is gone after the call' 'it is still there' "$tmp/rsync.sync-path"; else pass 'box.sh: the id'"'"'s temp dir is gone after the call'; fi
inlog 'box.sh: the sync id is not a file of the Mac tree' "$tmp/rsync.sync-path" '^/.*/box-sync\.[^/]*/\.sync-id$'
nofile 'box.sh: the Mac tree has no .sync-id' "$B/.sync-id"
if [ "$(tr '\n' ' ' < "$tmp/boxq.args")" = "run --card 3090 --class round --name x-echo-${id:0:6} --expected-s 90 -- " ]; then pass 'box.sh: the job words are run --card 3090 --class round --name x-echo-<id6> --expected-s 90 --'; else fail 'box.sh: the job words are run --card 3090 --class round --name x-echo-<id6> --expected-s 90 --' 'differ' "$tmp/boxq.args"; fi
inlog 'box.sh: the job holds the dir lock shared, 85 when it cannot' "$tmp/boxq.job" "^flock -s -n -E 8[5] /root/bloomery-queue/dir-x\.lock bash -c '"
if grep -Fq "[ \"\$(cat ~/repo/x/.sync-id 2>/dev/null)\" = $id ] ||" "$tmp/boxq.job"; then pass 'box.sh: the job checks the tree'"'"'s .sync-id against this call'"'"'s id first'; else fail 'box.sh: the job checks the tree'"'"'s .sync-id against this call'"'"'s id first' 'no such check' "$tmp/boxq.job"; fi
inlog 'box.sh: the job exits 80 on a stale tree' "$tmp/boxq.job" 'exit[ ]80; \}'
inlog 'box.sh: the job is the command as it runs directly' "$tmp/boxq.job" "^\} && cd ~/repo/x && .*echo hi'\$"
notinlog 'box.sh: the job has no guard' "$tmp/boxq.job" 'lease_guard|lease-probe'
notinlog 'box.sh: the ssh of a queued call has no guard and no command' "$tmp/ssh.log" 'lease_guard|source ~/bloomery-env'
inlog 'box.sh: the remote directory is made before the sync' "$tmp/ssh.log" '^-n stub mkdir -p ~/repo/x$'
# The exit code is the queue client's: its own codes and a job's pass through.
for code in 3 80 81 83 84 85 130 143; do
  rc_is "box.sh: a queued call exits with the client's $code" "$code" box env BLOOMERY_BOX_QUEUE=1 STUB_BOXQ_RC=$code "$B/tools/box.sh" true
done
# Predicted wall: unset is named and the job goes without one.
run 'box.sh: no prediction prints predicted unknown' 0 'box.sh: queue round card=3090 predicted unknown remote=~/repo/x$' \
  box env BLOOMERY_BOX_QUEUE=1 "$B/tools/box.sh" true
if grep -q -- '--expected-s' "$tmp/boxq.args"; then fail 'box.sh: no prediction submits without --expected-s' 'it was passed' "$tmp/boxq.args"; else pass 'box.sh: no prediction submits without --expected-s'; fi
for bad in 0 abc 1.5 -3 090; do
  run "box.sh: BLOOMERY_Q_PRED_S=$bad is refused" 64 'BLOOMERY_Q_PRED_S is whole seconds >= 1' box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_PRED_S=$bad "$B/tools/box.sh" true
  nofile "box.sh: the refused prediction $bad synced nothing" "$tmp/rsync.log"
done
# Class and name.
run 'box.sh: BLOOMERY_Q_CLASS reaches the job and the queue line' 0 'box.sh: queue lead card=3090 ' box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CLASS=lead "$B/tools/box.sh" true
inlog 'box.sh: --class lead' "$tmp/boxq.args" '^lead$'
run 'box.sh: BLOOMERY_Q_CLASS is a class' 64 'BLOOMERY_Q_CLASS is lead, round, sitting or fill, got .media.' box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CLASS=media "$B/tools/box.sh" true
rc_is 'box.sh: a command whose first word is a path' 0 box env BLOOMERY_BOX_QUEUE=1 "$B/tools/box.sh" './target/release/gate_x --a b'
inlog 'box.sh: the name takes the last part of the first word' "$tmp/boxq.args" '^x-gate_x-[0-9a-f]{6}$'
rc_is 'box.sh: a command whose first word is an assignment' 0 box env BLOOMERY_BOX_QUEUE=1 "$B/tools/box.sh" 'FOO=1 true'
inlog 'box.sh: the name holds only [A-Za-z0-9._-]' "$tmp/boxq.args" '^x-FOO_1-[0-9a-f]{6}$'
# A held dir lock: rsync fails (12), and the lock decides 83; a failure with the lock free or untestable is rsync's own.
run 'box.sh: a failed sync with the dir lock held ends 83, named' 83 'a job of this remote directory is running .*bloomery-queue/dir-x\.lock is held on stub.; its tree cannot be replaced under it' \
  box env BLOOMERY_BOX_QUEUE=1 STUB_RSYNC_RC=12 STUB_PROBE_RC=1 "$B/tools/box.sh" true
nofile 'box.sh: the 83 submitted no job' "$tmp/boxq.args"
inlog 'box.sh: the 83 probed the dir lock once' "$tmp/ssh.log" '^-n stub mkdir -p /root/bloomery-queue && flock -n -x /root/bloomery-queue/dir-x\.lock true$'
if [ -e "$(cat "$tmp/rsync.sync-path" 2> /dev/null || echo "$tmp/none")" ]; then fail 'box.sh: the 83 left the id'"'"'s temp dir' 'it is still there'; else pass 'box.sh: the 83 left no temp dir'; fi
rc_is 'box.sh: a failed sync with the dir lock free is rsync'"'"'s own code' 12 box env BLOOMERY_BOX_QUEUE=1 STUB_RSYNC_RC=12 STUB_PROBE_RC=0 "$B/tools/box.sh" true
rc_is 'box.sh: a failed sync whose probe cannot reach the box is rsync'"'"'s own code' 12 box env BLOOMERY_BOX_QUEUE=1 STUB_RSYNC_RC=12 STUB_PROBE_RC=255 "$B/tools/box.sh" true
rc_is 'box.sh: a direct call whose rsync fails keeps rsync'"'"'s code' 12 box env STUB_RSYNC_RC=12 STUB_PROBE_RC=1 "$B/tools/box.sh" true
notinlog 'box.sh: a direct call probes no lock' "$tmp/ssh.log" 'flock'
# The lane: BLOOMERY_CARD, or BLOOMERY_Q_CARD (any and none too); two that differ are refused, before any sync.
for lane in a6000 both; do
  rc_is "box.sh: BLOOMERY_CARD=$lane is the queue lane" 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_CARD=$lane "$B/tools/box.sh" true
  if [ "$(tr '\n' ' ' < "$tmp/boxq.args" | cut -d' ' -f1-3)" = "run --card $lane" ]; then pass "box.sh: --card $lane"; else fail "box.sh: --card $lane" 'differ' "$tmp/boxq.args"; fi
done
rc_is 'box.sh: BLOOMERY_Q_CARD=a6000 alone is the lane' 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CARD=a6000 "$B/tools/box.sh" true
inlog 'box.sh: the lane from BLOOMERY_Q_CARD' "$tmp/boxq.args" '^a6000$'
rc_is 'box.sh: BLOOMERY_CARD and BLOOMERY_Q_CARD that agree are one lane' 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_CARD=a6000 BLOOMERY_Q_CARD=a6000 "$B/tools/box.sh" true
run 'box.sh: BLOOMERY_CARD and BLOOMERY_Q_CARD that differ are refused' 64 "BLOOMERY_CARD is 'a6000' and BLOOMERY_Q_CARD is 'both': name one lane" \
  box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_CARD=a6000 BLOOMERY_Q_CARD=both "$B/tools/box.sh" true
nofile 'box.sh: the differing lanes synced nothing' "$tmp/rsync.log"
run 'box.sh: BLOOMERY_CARD=any is not a lane (only BLOOMERY_Q_CARD takes it)' 64 'BLOOMERY_CARD must be 3090, a6000 or both' box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_CARD=any "$B/tools/box.sh" true
run 'box.sh: BLOOMERY_Q_CARD is a lane' 64 'BLOOMERY_Q_CARD must be 3090, a6000, both, any or none, got .moon.' box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CARD=moon "$B/tools/box.sh" true
# `any`: the card is the daemon's choice, read when the job starts; `none`: no card pin beyond the default.
rc_is 'box.sh: BLOOMERY_Q_CARD=any' 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CARD=any "$B/tools/box.sh" true
inlog 'box.sh: --card any' "$tmp/boxq.args" '^any$'
inlog 'box.sh: any picks the card from $SIGNALBOX_CARD when the job starts' "$tmp/boxq.job" '\{ case "\$\{SIGNALBOX_CARD:-\}" in$'
inlog 'box.sh: any exports the daemon'"'"'s card as BLOOMERY_BOX_CARD' "$tmp/boxq.job" 'BLOOMERY_BOX_CARD="\$SIGNALBOX_CARD" && '
notinlog 'box.sh: any fixes no card on the Mac' "$tmp/boxq.job" 'BLOOMERY_BOX_CARD=(3090|a6000|any) '
cp "$tmp/boxq.job" "$Q/any.job"
rc_is 'box.sh: BLOOMERY_Q_CARD=none' 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_Q_CARD=none "$B/tools/box.sh" true
inlog 'box.sh: --card none' "$tmp/boxq.args" '^none$'
inlog 'box.sh: none keeps the default card pin and BLOOMERY_BOX_CARD=3090' "$tmp/boxq.job" 'BLOOMERY_BOX_CARD=3090 && '
notinlog 'box.sh: none picks no card' "$tmp/boxq.job" 'CUDA_VISIBLE_DEVICES|SIGNALBOX_CARD'
# A read never goes through the queue, whatever the queue's own variables say.
rc_is 'box.sh: a read under BLOOMERY_BOX_QUEUE=1 runs' 0 box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_BOX_READONLY=1 BLOOMERY_Q_CARD=any "$B/tools/box.sh" 'ps -e | wc -l'
nofile 'box.sh: a read under the queue submits no job' "$tmp/boxq.args"
nofile 'box.sh: a read under the queue syncs nothing' "$tmp/rsync.log"
inlog 'box.sh: a read under the queue is the direct read' "$tmp/ssh.log" '^stub source ~/bloomery-env\.sh && \{ :$'
notinlog 'box.sh: a read under the queue has no dir lock' "$tmp/ssh.log" 'bloomery-queue|flock'
run 'box.sh: a read under the queue still refuses a build' 64 "names 'cargo'" box env BLOOMERY_BOX_QUEUE=1 BLOOMERY_BOX_READONLY=1 "$B/tools/box.sh" 'cargo --version'

# The job's own text, run as the daemon runs it (`bash -c <text>`) in a sandbox: its HOME holds the env file and the remote
# directory, nvidia-smi and systemctl are stubs, and the dir lock is a file of this directory.
QH=$Q/home
mkdir -p "$QH/repo/x/tools/ref/models" "$Q/bin" "$Q/locks"
printf 'export CUDA_VISIBLE_DEVICES=GPU-default\n' > "$QH/bloomery-env.sh"
printf 'BLOOMERY_MODEL=prof\nMODEL=/m/a.gguf\nBLOOMERY_DATA=/d\nFIXTURE_FILE=\n' > "$QH/repo/x/tools/ref/ref-paths.sh"
printf 'V41_MODEL=/m/v.gguf\n' > "$QH/repo/x/tools/ref/models/deepseek41.sh"
printf '#!/bin/sh\ncase "$*" in *--query-gpu=uuid,name*) printf "GPU-A, NVIDIA RTX A6000\\nGPU-30, NVIDIA GeForce RTX 3090\\n" ;; esac\n' > "$Q/bin/nvidia-smi"
printf '#!/bin/sh\necho inactive\n' > "$Q/bin/systemctl"
chmod +x "$Q/bin/nvidia-smi" "$Q/bin/systemctl"
# qjob <env NAME=value…> -- <command>: box.sh's job text for the command, with the lock under $Q/locks, into $Q/job.sh.
qjob() {
  local envs=() w
  while [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  box env BLOOMERY_BOX_QUEUE=1 ${envs[@]+"${envs[@]}"} "$B/tools/box.sh" "$@" > /dev/null 2>&1
  sed "s#/root/bloomery-queue#$Q/locks#g" "$tmp/boxq.job" > "$Q/job.sh"
  QID=$(cat "$tmp/rsync.sync-id")
}
# qrun <name> <rc> <pattern> <NAME=value…>: the job text in the sandbox with those variables; the tree's .sync-id is $QID.
qrun() {
  local name=$1 want=$2 pat=$3
  shift 3
  printf '%s\n' "$QID" > "$QH/repo/x/.sync-id"
  run "$name" "$want" "$pat" env -u SIGNALBOX_CARD HOME="$QH" PATH="$Q/bin:$PATH" "$@" bash -c "$(cat "$Q/job.sh")"
}
# qrc <name> <rc> <NAME=value…>: the same, the code only (a job that prints nothing).
qrc() {
  local name=$1 want=$2
  shift 2
  printf '%s\n' "$QID" > "$QH/repo/x/.sync-id"
  rc_is "$name" "$want" env -u SIGNALBOX_CARD HOME="$QH" PATH="$Q/bin:$PATH" "$@" bash -c "$(cat "$Q/job.sh")"
}
SHOW='echo "dev=$CUDA_VISIBLE_DEVICES card=$BLOOMERY_BOX_CARD sbc=${SIGNALBOX_CARD-}"; exit 3'
qjob -- "$SHOW"
qrun 'box.sh job: the command runs and its code is the job'"'"'s' 3 '^dev=GPU-default card=3090 sbc=$'
qjob BLOOMERY_CARD=a6000 -- "$SHOW"
qrun 'box.sh job: lane a6000 pins the A6000' 3 '^dev=GPU-A card=a6000 sbc=$'
qjob BLOOMERY_CARD=both -- "$SHOW"
qrun 'box.sh job: lane both pins the 3090 then the A6000' 3 '^dev=GPU-30,GPU-A card=both sbc=$'
qjob BLOOMERY_Q_CARD=any -- "$SHOW"
qrun 'box.sh job: any on the daemon'"'"'s a6000 pins the A6000' 3 '^dev=GPU-A card=a6000 sbc=a6000$' SIGNALBOX_CARD=a6000
qrun 'box.sh job: any on the daemon'"'"'s 3090 keeps the default pin' 3 '^dev=GPU-default card=3090 sbc=3090$' SIGNALBOX_CARD=3090
qrun 'box.sh job: any with no SIGNALBOX_CARD is a named 70, the command not run' 70 'a job of lane any starts with SIGNALBOX_CARD 3090 or a6000 \(the daemon sets it\), got ..$'
qrun 'box.sh job: any on a lane that is neither card is a named 70' 70 'got .both.$' SIGNALBOX_CARD=both
qjob BLOOMERY_Q_CARD=none -- "$SHOW"
qrun 'box.sh job: none keeps the default pin' 3 '^dev=GPU-default card=3090 sbc=none$' SIGNALBOX_CARD=none
# The stale-tree check: another sync (a different id, or none) between the submit and the start ends 80 before the command.
qjob -- "touch $Q/ran; echo ran"
rm -f "$Q/ran"
printf 'other\n' > "$QH/repo/x/.sync-id"
run 'box.sh job: a tree synced again since the submit ends 80, named' 80 'box.sh: ~/repo/x was synced again after this job was queued \(stale tree\)' \
  env HOME="$QH" PATH="$Q/bin:$PATH" bash -c "$(cat "$Q/job.sh")"
nofile 'box.sh job: the stale tree ran nothing' "$Q/ran"
rm -f "$QH/repo/x/.sync-id"
run 'box.sh job: a tree with no .sync-id ends 80' 80 'stale tree' env HOME="$QH" PATH="$Q/bin:$PATH" bash -c "$(cat "$Q/job.sh")"
nofile 'box.sh job: the tree with no .sync-id ran nothing' "$Q/ran"
qrun 'box.sh job: the same id runs the command' 0 '^ran$'
if [ -e "$Q/ran" ]; then pass 'box.sh job: the command ran'; else fail 'box.sh job: the command ran' 'no marker'; fi
# The dir lock: held exclusive (a sync in flight) the job ends 85, named by the code only, and runs nothing; held shared (another job of the
# directory) it runs beside it. Its own lock stays held for the job's life, children included.
rm -f "$Q/ran"
if bg_lock -x "$Q/locks/dir-x.lock"; then
  qrc 'box.sh job: the dir lock held exclusive ends 85' 85
  nofile 'box.sh job: the 85 ran nothing' "$Q/ran"
  bg_end
else
  fail 'box.sh job: the dir lock held exclusive' 'the background lock was not taken'
fi
if bg_lock -s "$Q/locks/dir-x.lock"; then
  qrun 'box.sh job: the dir lock held shared by another job lets this one run' 0 '^ran$'
  bg_end
else
  fail 'box.sh job: the dir lock held shared' 'the background lock was not taken'
fi
# A command with quotes, dollars, newlines and a semicolon crosses the job's quoting whole.
qjob -- 'printf "%s|" "it'"'"'s" '"'"'$HOME `x`'"'"'; printf "a\nb\n"; echo "$((1+2))"'
qrun 'box.sh job: quotes, dollars and newlines in the command cross whole' 0 "^it's[|]\\\$HOME \`x\`[|]a$"
inlog 'box.sh job: the quoted command ends the output the same way' "$tmp/out" '^3$'

# Local mode (BLOOMERY_BOX_LOCAL=1): the same command runs here, in the tree, through bash; nothing reaches ssh or rsync. The tree
# gets the files the command reads (the guard's lease-probe.sh, the profiles), the environment file is a stub that says it ran,
# the locks live in a directory of the test's own, and a stub uname says Linux (STUB_UNAME names another system).
mkdir -p "$B/tools/ref/models" "$tmp/home" "$tmp/uname"
cp "$ROOT/tools/ref/lease-probe.sh" "$ROOT/tools/ref/ref-paths.sh" "$B/tools/ref/"
cp "$ROOT"/tools/ref/models/*.sh "$B/tools/ref/models/"
printf 'echo "[env] sourced"\n' > "$tmp/local-env.sh"
printf '#!/bin/sh\necho "${STUB_UNAME:-Linux}"\n' > "$tmp/uname/uname"
chmod +x "$tmp/uname/uname"
L=$tmp/locks-local
SHOW='echo "pwd=$PWD card=$BLOOMERY_BOX_CARD locks=$BLOOMERY_LOCK_DIR oxide=${CUDA_OXIDE_BACKEND-unset} data=$BLOOMERY_DATA jobs=${CARGO_BUILD_JOBS-unset}"'
boxl() {
  box env -u CUDA_OXIDE_BACKEND -u CARGO_BUILD_JOBS PATH="$tmp/uname:$tmp/stub:$PATH" BLOOMERY_BOX_LOCAL=1 \
    BLOOMERY_ENV_FILE="$tmp/local-env.sh" BLOOMERY_LOCK_DIR="$L" BLOOMERY_DATA="$tmp/data" "$@"
}
unreached() {
  if [ -s "$tmp/ssh.log" ] || [ -s "$tmp/rsync.log" ]; then cat "$tmp/ssh.log" "$tmp/rsync.log" > "$tmp/both.log"; fail "$1" 'ssh or rsync ran' "$tmp/both.log"; else pass "$1"; fi
}
if [ -e /root/bloomery-cpu.lock ]; then
  run 'box.sh: local mode on the box is refused' 64 'BLOOMERY_BOX_LOCAL=1 on the machine whose lease is /root/bloomery-cpu\.lock' boxl "$B/tools/box.sh" "$SHOW"
  unreached 'box.sh: the refusal on the box reached no ssh'
else
  run 'box.sh: local mode runs the command here, after the environment file' 0 '^\[env\] sourced$' boxl "$B/tools/box.sh" "$SHOW"
  has 'box.sh: local mode runs it in the tree, the card local, the locks and data passed, no oxide backend and no build jobs set' \
    "^pwd=$B card=local locks=$L oxide=unset data=$tmp/data jobs=unset\$"
  unreached 'box.sh: local mode reaches no ssh and no rsync'
  if [ -d "$L" ]; then pass 'box.sh: local mode made its lock directory'; else fail 'box.sh: local mode made its lock directory' "no $L"; fi
  printf 'export CUDA_OXIDE_BACKEND=/x/backend.so\n' > "$tmp/oxide-env.sh"
  # A cargo oxide command still takes the installed cargo-oxide check in local mode: a stub cargo at the pin passes it.
  mkdir -p "$tmp/oxpin"
  printf '#!/bin/sh\necho "cargo-oxide v0.2.1 (https://github.com/midagedev/cuda-oxide.git?rev=x#%s):"\n' "${PIN:0:8}" > "$tmp/oxpin/cargo"
  chmod +x "$tmp/oxpin/cargo"
  run 'box.sh: local mode unsets the environment file'"'"'s CUDA_OXIDE_BACKEND and adds no backend check to cargo oxide' 0 'oxide=unset data=' \
    boxl PATH="$tmp/oxpin:$tmp/uname:$tmp/stub:$PATH" BLOOMERY_ENV_FILE="$tmp/oxide-env.sh" "$B/tools/box.sh" "echo 'cargo oxide build' && $SHOW"
  printf 'export BLOOMERY_LOCK_DIR=%s/elsewhere\n' "$tmp" > "$tmp/lockdir-env.sh"
  run "box.sh: local mode refuses an environment file that moves the lock directory after the guard" 64 \
    'the environment file or BLOOMERY_BOX_ENV set BLOOMERY_LOCK_DIR after local mode.s guard probed the lock directory' \
    boxl BLOOMERY_ENV_FILE="$tmp/lockdir-env.sh" "$B/tools/box.sh" "$SHOW"
  if grep -q '^pwd=' "$tmp/out"; then fail 'box.sh: that refusal ran no command' 'the command ran' "$tmp/out"; else pass 'box.sh: that refusal ran no command'; fi
  run 'box.sh: local mode runs BLOOMERY_CARD=a6000 on the local card' 0 ' card=local ' boxl BLOOMERY_CARD=a6000 "$B/tools/box.sh" "$SHOW"
  run 'box.sh: local mode refuses BLOOMERY_CARD=both' 64 "BLOOMERY_CARD=both puts the box's two cards in view, and BLOOMERY_BOX_LOCAL=1 runs on what this machine shows" \
    boxl BLOOMERY_CARD=both "$B/tools/box.sh" "$SHOW"
  if grep -q '^pwd=' "$tmp/out"; then fail 'box.sh: the refusal of both ran no command' 'the command ran' "$tmp/out"; else pass 'box.sh: the refusal of both ran no command'; fi
  L2="$tmp/lock dir;x"
  mkdir -p "$L2" && touch "$L2/bloomery-aa-hold"
  run 'box.sh: the local guard reads the holds in the lock directory, a path with a space and a ; whole' 75 "hold $L2/bloomery-aa-hold owner=aa " \
    boxl BLOOMERY_LOCK_DIR="$L2" BLOOMERY_BOX_WAIT=0 "$B/tools/box.sh" "$SHOW"
  if grep -q '^pwd=' "$tmp/out"; then fail 'box.sh: the local guard ran no command while the hold was up' 'the command ran' "$tmp/out"; else
    pass 'box.sh: the local guard ran no command while the hold was up'; fi
  run 'box.sh: local mode with no environment file: 69, named' 69 "sources the environment file BLOOMERY_ENV_FILE .*'$tmp/no-env\.sh' is not there" \
    boxl BLOOMERY_ENV_FILE="$tmp/no-env.sh" "$B/tools/box.sh" "$SHOW"
  run 'box.sh: local mode keeps its locks under ~/.cache/bloomery/locks by default' 0 "locks=$tmp/home/\\.cache/bloomery/locks " \
    boxl BLOOMERY_LOCK_DIR= HOME="$tmp/home" "$B/tools/box.sh" "$SHOW"
  run "box.sh: local mode refuses the lock directory /root, the box's" 64 "BLOOMERY_LOCK_DIR=/root is the box's lock directory: local mode's guard would make /root/bloomery-cpu\\.lock" \
    boxl BLOOMERY_LOCK_DIR=/root "$B/tools/box.sh" "$SHOW"
  run 'box.sh: local mode refuses a relative lock directory, which two trees would read apart' 64 "BLOOMERY_LOCK_DIR 'rel-locks' is relative" \
    boxl BLOOMERY_LOCK_DIR=rel-locks "$B/tools/box.sh" "$SHOW"
  printf 'export BLOOMERY_ARCH=sm_121\n' > "$tmp/arch-env.sh"
  run "box.sh: local mode refuses an environment file's BLOOMERY_ARCH that the recipe did not expand" 64 \
    'the environment file set BLOOMERY_ARCH=sm_121, and the recipe expanded sm_86' boxl BLOOMERY_ENV_FILE="$tmp/arch-env.sh" "$B/tools/box.sh" "$SHOW"
  sel='echo "model=${BLOOMERY_MODEL-unset} card=${BLOOMERY_CARD-unset} local=${BLOOMERY_BOX_LOCAL-unset} env=${BLOOMERY_ENV_FILE-unset}'
  sel+=' queue=${BLOOMERY_BOX_QUEUE-unset} class=${BLOOMERY_Q_CLASS-unset} qcard=${BLOOMERY_Q_CARD-unset} pred=${BLOOMERY_Q_PRED_S-unset}'
  sel+=' budget=${BLOOMERY_CARD_BUDGET-unset} fixture=${BLOOMERY_FIXTURE_MODEL-unset} box=${BLOOMERY_BOX-unset} remote=${BLOOMERY_REMOTE-unset}'
  sel+=' wait=${BLOOMERY_BOX_WAIT-unset} owner=${BLOOMERY_HOLD_OWNER-unset} boxenv=${BLOOMERY_BOX_ENV-unset} ro=${BLOOMERY_BOX_READONLY-unset}'
  sel+=' profile=${BLOOMERY_REF_MODEL_PROFILE-unset} lease=${BLOOMERY_LEASE_LOCK-unset} holds=${BLOOMERY_LEASE_HOLDS-unset}"'
  run "box.sh: the command gets none of what box.sh read on this side, its own outputs only as it exports them, as on the box" 0 \
    '^model=unset card=unset local=unset env=unset queue=unset class=unset qcard=unset pred=unset budget=unset fixture=unset box=unset remote=unset wait=unset owner=unset boxenv=unset ro=unset profile=deepseek2 lease=unset holds=unset$' \
    boxl BLOOMERY_MODEL=deepseek2 BLOOMERY_CARD=a6000 BLOOMERY_BOX_QUEUE=0 BLOOMERY_Q_CLASS=lead BLOOMERY_Q_CARD=a6000 \
    BLOOMERY_Q_PRED_S=90 BLOOMERY_CARD_BUDGET=12345 BLOOMERY_FIXTURE_MODEL=/m/x.gguf BLOOMERY_BOX=elsewhere BLOOMERY_REMOTE='~/x' \
    BLOOMERY_BOX_WAIT=7 BLOOMERY_HOLD_OWNER=zz BLOOMERY_BOX_ENV=BLOOMERY_PREHEAT=0 BLOOMERY_BOX_READONLY=0 \
    BLOOMERY_REF_MODEL_PROFILE=qwen4exp BLOOMERY_LEASE_LOCK="$tmp/other.lock" BLOOMERY_LEASE_HOLDS="$tmp/other-*-hold" "$B/tools/box.sh" "$sel"
  run "box.sh: just box-gc's script refuses local mode by name" 64 "box-gc: BLOOMERY_BOX_CARD=local is tools/box\\.sh's local mode" \
    boxl BLOOMERY_BOX_READONLY=1 "$B/tools/box.sh" 'bash -s -- --kill' < "$ROOT/tools/box-gc.sh"
fi
run 'box.sh: local mode refuses the box queue' 64 "BLOOMERY_BOX_QUEUE=1 makes the command a job of the box's queue, and BLOOMERY_BOX_LOCAL=1 runs it here" \
  boxl BLOOMERY_BOX_QUEUE=1 "$B/tools/box.sh" "$SHOW"
if grep -q '^pwd=' "$tmp/out"; then fail 'box.sh: the refusal of the queue ran no command' 'the command ran' "$tmp/out"; else pass 'box.sh: the refusal of the queue ran no command'; fi
nofile 'box.sh: the refusal of the queue runs no queue client' "$tmp/boxq.args"
run 'box.sh: local mode refuses a host that is not Linux' 64 "BLOOMERY_BOX_LOCAL=1 runs the box's commands here, and they need Linux \\(this is Darwin\\)" \
  boxl STUB_UNAME=Darwin "$B/tools/box.sh" "$SHOW"
run 'box.sh: BLOOMERY_BOX_LOCAL takes 1 or 0' 64 'BLOOMERY_BOX_LOCAL is 1 \(run here\) or 0' box env BLOOMERY_BOX_LOCAL=yes "$B/tools/box.sh" true

# tools/gpu-gate.sh takes the gate lock of the card box.sh put in view (BLOOMERY_BOX_CARD): a card named
# against it, and `both` without both cards in view, are refused before any lock is opened. The paths
# that take a /root gate lock are the box's to show, not these tests'; local mode's lock is in a directory of the test's.
G=$tmp/gatetree
mkdir -p "$G/tools/ref" "$G/target/release"
cp "$ROOT/tools/gpu-gate.sh" "$ROOT/tools/gate-bound.sh" "$G/tools/"
cp "$ROOT/tools/ref/lease-probe.sh" "$G/tools/ref/"
printf '#!/bin/sh\nexit 0\n' > "$G/target/release/x"
chmod +x "$G/target/release/x"
gg() { (cd "$G" && env -u BLOOMERY_GATE_CARD -u BLOOMERY_BOX_CARD -u BLOOMERY_GATE_BOUND -u BLOOMERY_LOCK_DIR "$@" bash tools/gpu-gate.sh x); }
run 'gpu-gate.sh: both cards in view refuse the 3090 lock alone' 64 'box.sh put both cards in view \(BLOOMERY_CARD=both\), and BLOOMERY_GATE_CARD=3090 names another lock' \
  gg BLOOMERY_BOX_CARD=both BLOOMERY_GATE_CARD=3090
run 'gpu-gate.sh: both cards in view refuse any' 64 'box.sh put both cards in view' gg BLOOMERY_BOX_CARD=both BLOOMERY_GATE_CARD=any
run 'gpu-gate.sh: the A6000 in view refuses the 3090 lock' 64 'box.sh put the A6000 in view \(BLOOMERY_CARD=a6000\)' \
  gg BLOOMERY_BOX_CARD=a6000 BLOOMERY_GATE_CARD=3090
run 'gpu-gate.sh: both needs both cards in view' 64 "run it under box.sh's BLOOMERY_CARD=both" gg BLOOMERY_GATE_CARD=both
run 'gpu-gate.sh: BLOOMERY_BOX_CARD is a card box.sh picks' 64 'BLOOMERY_BOX_CARD is 3090, a6000, both or local' gg BLOOMERY_BOX_CARD=two
run "gpu-gate.sh: local is box.sh's choice, never a gate card" 64 "BLOOMERY_GATE_CARD is 3090, a6000, any or both, got 'local'" \
  gg BLOOMERY_GATE_CARD=local
# The box (its lease file): box.sh's local card and another lock directory would hold none of the box's locks. Elsewhere local mode
# keeps its one lock under a lock directory that is not /root, and the path is opened whole.
if [ -e /root/bloomery-cpu.lock ]; then
  run "gpu-gate.sh: on the box, box.sh's local card is refused" 64 'BLOOMERY_BOX_CARD=local on the machine whose lease is /root/bloomery-cpu\.lock' \
    gg BLOOMERY_BOX_CARD=local
  run 'gpu-gate.sh: on the box, a lock directory other than /root is refused' 64 "BLOOMERY_LOCK_DIR=$tmp/ld on the machine whose lease is /root/bloomery-cpu\\.lock" \
    gg BLOOMERY_LOCK_DIR="$tmp/ld"
else
  run "gpu-gate.sh: box.sh's local mode refuses both" 64 "BLOOMERY_GATE_CARD=both names the box's two cards" \
    gg BLOOMERY_BOX_CARD=local BLOOMERY_LOCK_DIR="$tmp/ld" BLOOMERY_GATE_CARD=both
  run "gpu-gate.sh: box.sh's local mode refuses the lock directory /root" 64 'keeps its lock under BLOOMERY_LOCK_DIR, and that is /root \(unset: /root\)' \
    gg BLOOMERY_BOX_CARD=local
  run 'gpu-gate.sh: BLOOMERY_LOCK_DIR holds the gate locks' 69 "cannot open $tmp/no-such-dir/bloomery-gate\\.lock" gg BLOOMERY_LOCK_DIR="$tmp/no-such-dir"
  run "gpu-gate.sh: box.sh's local mode opens its one lock there, the path whole" 69 "cannot open $tmp/no such;dir/bloomery-gate-local\\.lock .*local mode" \
    gg BLOOMERY_BOX_CARD=local BLOOMERY_LOCK_DIR="$tmp/no such;dir"
fi
run 'gpu-gate.sh: BLOOMERY_GATE_CARD is a card' 64 'BLOOMERY_GATE_CARD is 3090, a6000, any or both' gg BLOOMERY_GATE_CARD=all
if [ -w /root ]; then
  echo "skip gpu-gate.sh lock-open test: /root is writable here (the box), where it would take the real gate lock"
else
  run 'gpu-gate.sh: a lock file it cannot open ends at once, by name' 69 'cannot open /root/bloomery-gate\.lock' gg
fi

# tools/gate.sh --oxide builds for the arch tools/arch.sh reads (BLOOMERY_ARCH, sm_86 unset) and refuses a value that is no arch by
# name before cargo runs; a call without --oxide never reads it. The stub timeout drops its two words and the stub cargo prints its.
GS=$tmp/gateshtree
mkdir -p "$GS/tools" "$GS/stub"
cp "$ROOT/tools/gate.sh" "$ROOT/tools/gate-bound.sh" "$ROOT/tools/arch.sh" "$GS/tools/"
printf '#!/bin/sh\nshift 2\nexec "$@"\n' > "$GS/stub/timeout"
printf '#!/bin/sh\necho "cargo $*"\n' > "$GS/stub/cargo"
chmod +x "$GS/stub/timeout" "$GS/stub/cargo"
gs() { (cd "$GS" && env -u BLOOMERY_ARCH -u BLOOMERY_GATE_BOUND PATH="$GS/stub:$PATH" "$@" bash tools/gate.sh "${GS_ARGS[@]}"); }
GS_ARGS=(--oxide -p x --release --lib)
run 'gate.sh: --oxide builds for sm_86 when BLOOMERY_ARCH is unset' 0 '^cargo oxide test --arch sm_86 -- -p x --release --lib$' gs
run 'gate.sh: --oxide builds for BLOOMERY_ARCH' 0 '^cargo oxide test --arch sm_121 -- -p x --release --lib$' gs BLOOMERY_ARCH=sm_121
run 'gate.sh: --oxide refuses a BLOOMERY_ARCH that is no arch, by name' 64 "gate.sh: BLOOMERY_ARCH is a CUDA arch.*got 'sm121'" gs BLOOMERY_ARCH=sm121
GS_ARGS=(-p x --release --lib)
run 'gate.sh: a plain cargo test never reads BLOOMERY_ARCH' 0 '^cargo test -p x --release --lib$' gs BLOOMERY_ARCH=sm121

# card-precheck.sh: lease_card before the build, with the recipe's own card or the environment's.
pre() {
  local name=$1 want=$2 pat=$3
  shift 3
  run "$name" "$want" "$pat" env -u BLOOMERY_LEASE_CARD -u BLOOMERY_DRY -u ROUNDS -u BLOOMERY_AB_ROUNDS "$@"
}
pre 'precheck without a card refuses before any build' 66 'refused CARD_ABSENT \(rc 66\): BLOOMERY_LEASE_CARD is empty' \
  "$T/tools/ref/card-precheck.sh"
pre 'precheck passes the environment card' 0 '^\[lease\] card: ok kind=ab h = 0\.875 % at 5 rounds' \
  BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_AB_ROUNDS=5 "$T/tools/ref/card-precheck.sh"
pre 'precheck checks an ab card at BLOOMERY_AB_ROUNDS' 68 'refused CARD_UNDER_RULER' \
  BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_AB_ROUNDS=3 "$T/tools/ref/card-precheck.sh"
pre 'precheck takes the recipe card' 0 '^\[lease\] card: ok kind=exclusive minutes=5\.\.15' \
  "$T/tools/ref/card-precheck.sh" docs/cards/exclusive.card
pre 'precheck refuses two cards' 64 'name two cards' BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card \
  "$T/tools/ref/card-precheck.sh" docs/cards/exclusive.card
pre 'precheck passes a dry run by name' 0 '^card-precheck: BLOOMERY_DRY is set' BLOOMERY_DRY=1 "$T/tools/ref/card-precheck.sh"
pre 'this tree: precheck with the exact-ref card' 0 '^\[lease\] card: ok kind=exclusive minutes=1\.\.3 ' \
  "$ROOT/tools/ref/card-precheck.sh" docs/cards/exact-ref.card

# The witness fields that ask the cards: a failing nvidia-smi (a card off the bus) is named in its field,
# `<field>: unavailable (rc N)`, and the runner goes on under set -euo pipefail.
mkdir -p "$tmp/smi"
printf '#!/bin/sh\necho "NVIDIA-SMI has failed (test stub)" >&2\nexit 3\n' > "$tmp/smi/nvidia-smi"
chmod +x "$tmp/smi/nvidia-smi"
for fields in 'head indent gpus' 'head indent stage0-gpus' 'head indent stage0-apps' 'head-load indent gpu-apps'; do
  run "witness ($fields) with a failing nvidia-smi goes on" 0 '^\[after\] the runner goes on$' \
    env PATH="$tmp/smi:$PATH" GPU_3090=0 bash -c 'set -euo pipefail; source "$1"; read -r -a WITNESS <<< "$2"; witness pre; echo "[after] the runner goes on"' \
    _ "$T/tools/ref/lease.sh" "$fields"
  has "witness ($fields) names the failure" '(gpus|stage0-gpus|compute-apps-3090|gpu-apps): unavailable \(rc 3\)$|gpu=unavailable \(rc 3\)$'
done

# tools/ref/cards.sh: the two cards' UUIDs resolved from their names with one nvidia-smi query —
# the repository is public and carries no UUID. A name that matches no card, or two, leaves its
# variable empty with the reason in CARDS_ERROR, and sourcing never exits: a runner that needs
# only the other card still runs. A fake nvidia-smi on PATH answers each case (the made-up UUIDs
# depth-stub-cards.sh uses, never the box's).
FAKE_3090=GPU-11111111-1111-1111-1111-111111111111
FAKE_A6000=GPU-00000000-0000-0000-0000-000000000000
C=$tmp/cards
mknvidia() { # <dir>: a stub nvidia-smi that prints the "name, uuid" lines stdin gives
  mkdir -p "$1"
  { echo '#!/bin/sh'; echo "cat << 'EOF'"; cat; echo 'EOF'; } > "$1/nvidia-smi"
  chmod +x "$1/nvidia-smi"
}
mknvidia "$C/both" << EOF
NVIDIA GeForce RTX 3090, $FAKE_3090
NVIDIA RTX A6000, $FAKE_A6000
EOF
mknvidia "$C/no3090" << EOF
NVIDIA RTX A6000, $FAKE_A6000
EOF
mknvidia "$C/twice" << EOF
NVIDIA GeForce RTX 3090, $FAKE_3090
NVIDIA GeForce RTX 3090, $FAKE_3090
NVIDIA RTX A6000, $FAKE_A6000
EOF
mkdir -p "$C/failed"
{ echo '#!/bin/sh'; echo 'echo "nvidia-smi: driver not loaded (stub)" >&2'; echo 'exit 9'; } > "$C/failed/nvidia-smi"
chmod +x "$C/failed/nvidia-smi"
cards_src='source "$1" && printf "3090=[%s] a6000=[%s] err=[%s]\n" "$GPU_3090" "$GPU_A6000" "$CARDS_ERROR"'
cards() { # <name> <rc> <pattern> <dir>: the tree's cards.sh sourced with <dir>'s stub nvidia-smi on PATH
  run "$1" "$2" "$3" env PATH="$4:$PATH" bash -c "$cards_src" _ "$ROOT/tools/ref/cards.sh"
}
cards 'cards: both names resolve' 0 "^3090=\[$FAKE_3090\] a6000=\[$FAKE_A6000\] err=\[\]$" "$C/both"
cards 'cards: the 3090 missing leaves its variable empty, named' 0 \
  "^3090=\[\] a6000=\[$FAKE_A6000\] err=\[.*'NVIDIA GeForce RTX 3090' matched 0 cards\]$" "$C/no3090"
cards 'cards: a name twice leaves that card empty, named' 0 \
  "^3090=\[\] a6000=\[$FAKE_A6000\] err=\[.*'NVIDIA GeForce RTX 3090' matched 2 cards\]$" "$C/twice"
cards 'cards: a failing nvidia-smi leaves both empty, named' 0 \
  '^3090=\[\] a6000=\[\] err=\[nvidia-smi --query-gpu=name,uuid failed \(rc 9\): nvidia-smi: driver not loaded \(stub\)\]$' "$C/failed"
# No nvidia-smi anywhere on PATH (bash by absolute path: the stub PATH has no tools in it).
run 'cards: no nvidia-smi on PATH leaves both empty, named' 0 \
  '^3090=\[\] a6000=\[\] err=\[nvidia-smi is not on PATH\]$' \
  env PATH="$C/none" /bin/bash -c "$cards_src" _ "$ROOT/tools/ref/cards.sh"

# tools/ref/timing-card.sh's local mode (BLOOMERY_BOX_CARD=local), sourced under set -euo pipefail by a runner
# that opts in: the timing card read by UUID with one query, never by name, each refusal in TIMING_CARDS_WHY
# (lease_take ends the run on it, 64), the Xid reader's journal check, the lost card and arm-place.sh's
# placement refusal. The stub nvidia-smi lists the UUIDs STUB_UUIDS gives (STUB_UUID_RC fails the list,
# STUB_UUID_WARN adds a warning line), answers every card's name as a GB10's and none once STUB_GONE is set;
# the stub journalctl has one kernel line (STUB_JOURNAL=fail fails it, empty shows none, warn none and a
# warning on stderr). The box refuses local mode, so there that refusal is the one case.
FAKE_GB10=GPU-22222222-2222-2222-2222-222222222222
mkdir -p "$C/local"
cat > "$C/local/nvidia-smi" << 'SMI'
#!/bin/sh
case "$*" in
  *--query-gpu=uuid\ *)
    [ -z "$STUB_UUID_RC" ] || { echo 'NVIDIA-SMI has failed (stub)'; exit "$STUB_UUID_RC"; }
    [ -z "$STUB_UUID_WARN" ] || echo 'WARNING: infoROM is corrupted (stub)'
    printf '%s\n' $STUB_UUIDS
    exit 0
    ;;
esac
[ -z "$STUB_GONE" ] || { echo 'GPU is lost (stub)' >&2; exit 15; }
echo 'NVIDIA GB10'
SMI
cat > "$C/local/journalctl" << 'JCTL'
#!/bin/sh
case $STUB_JOURNAL in
  fail) echo 'journalctl: the stub fails' >&2; exit 1 ;;
  empty) exit 0 ;;
  warn) echo 'No journal files were opened due to insufficient permissions (stub).' >&2; exit 0 ;;
esac
echo '1790428800.000000 ws kernel: Linux version (stub)'
JCTL
printf '#!/bin/sh\nshift 2\nexec "$@"\n' > "$C/local/timeout"
chmod +x "$C/local/nvidia-smi" "$C/local/journalctl" "$C/local/timeout"
locx() { # <name> <rc> <pattern> <commands after the source> [NAME=value…]: the tree's timing-card.sh in local mode
  local name=$1 want=$2 pat=$3 body=$4
  shift 4
  run "$name" "$want" "$pat" env -u BLOOMERY_TIMING_GPU -u BLOOMERY_TIMING_CARDS -u BLOOMERY_DRAFT -u STUB_UUID_RC -u STUB_UUID_WARN \
    -u STUB_GONE -u STUB_JOURNAL PATH="$C/local:$PATH" BLOOMERY_BOX_CARD=local BLOOMERY_LOCK_DIR="$tmp/ld" TIMING_LOCAL_RUNNER=1 \
    STUB_UUIDS="$FAKE_GB10" "$@" bash -c "set -euo pipefail; source \"\$1\"; $body" _ "$ROOT/tools/ref/timing-card.sh"
}
loc() { # <name> <pattern> [NAME=value…]: the card, CUDA_VISIBLE_DEVICES and the refusal after the source
  locx "$1" 0 "$2" 'printf "gpu=[%s] cvd=[%s] local=[%s] why=[%s]\n" "$TIMING_GPU" "$CUDA_VISIBLE_DEVICES" "$TIMING_LOCAL" "$TIMING_CARDS_WHY"' "${@:3}"
}
if [ -e /root/bloomery-cpu.lock ]; then
  loc 'timing-card.sh local: refused on the box' '^gpu=\[\] cvd=\[\] local=\[1\] why=\[BLOOMERY_BOX_CARD=local on the machine whose lease is /root/bloomery-cpu\.lock \(the box\): '
else
  loc 'timing-card.sh local: the one card listed is the timing card' "^gpu=\\[$FAKE_GB10\\] cvd=\\[$FAKE_GB10\\] local=\\[1\\] why=\\[\\]\$"
  loc 'timing-card.sh local: a warning beside the list is no card' "^gpu=\\[$FAKE_GB10\\] cvd=\\[$FAKE_GB10\\] local=\\[1\\] why=\\[\\]\$" STUB_UUID_WARN=1
  loc 'timing-card.sh local: a runner that does not opt in is refused' \
    '^gpu=\[\] cvd=\[\] local=\[1\] why=\[BLOOMERY_BOX_CARD=local, and _ has no local mode \(a runner opts in with TIMING_LOCAL_RUNNER=1\): ' \
    TIMING_LOCAL_RUNNER=
  has 'timing-card.sh local: the refusal line is tagged [timing-card]' '^\[timing-card\] refused: BLOOMERY_BOX_CARD=local, and _ has no local mode '
  loc 'timing-card.sh local: a failing nvidia-smi is refused' \
    '^gpu=\[\] cvd=\[\] local=\[1\] why=\[local mode: nvidia-smi --query-gpu=uuid failed \(rc 9\): NVIDIA-SMI has failed \(stub\)\]$' STUB_UUID_RC=9
  loc 'timing-card.sh local: no card listed is refused' '^gpu=\[\] cvd=\[\] local=\[1\] why=\[local mode: nvidia-smi --query-gpu=uuid lists no card\]$' \
    STUB_UUIDS=
  loc 'timing-card.sh local: two cards and no BLOOMERY_TIMING_GPU are refused' \
    "^gpu=\\[\\] cvd=\\[\\] local=\\[1\\] why=\\[local mode: nvidia-smi lists 2 cards \\($FAKE_GB10 $FAKE_A6000\\): name one by UUID in BLOOMERY_TIMING_GPU\\]\$" \
    STUB_UUIDS="$FAKE_GB10 $FAKE_A6000"
  loc 'timing-card.sh local: BLOOMERY_TIMING_GPU picks a listed card' "^gpu=\\[$FAKE_A6000\\] cvd=\\[$FAKE_A6000\\] local=\\[1\\] why=\\[\\]\$" \
    STUB_UUIDS="$FAKE_GB10 $FAKE_A6000" BLOOMERY_TIMING_GPU="$FAKE_A6000"
  loc 'timing-card.sh local: a BLOOMERY_TIMING_GPU nvidia-smi does not list is refused' \
    "^gpu=\\[\\] cvd=\\[\\] local=\\[1\\] why=\\[local mode: BLOOMERY_TIMING_GPU=$FAKE_3090 is not a card nvidia-smi lists \\($FAKE_GB10\\)\\]\$" \
    BLOOMERY_TIMING_GPU="$FAKE_3090"
  loc 'timing-card.sh local: BLOOMERY_TIMING_CARDS is refused' \
    '^gpu=\[\] cvd=\[\] local=\[1\] why=\[BLOOMERY_TIMING_CARDS=a6000\+3090 and BLOOMERY_BOX_CARD=local: ' BLOOMERY_TIMING_CARDS=a6000+3090
  loc 'timing-card.sh local: the lock directory /root is refused' \
    "^gpu=\\[\\] cvd=\\[\\] local=\\[1\\] why=\\[BLOOMERY_BOX_CARD=local keeps the lease under BLOOMERY_LOCK_DIR, and that is /root " \
    BLOOMERY_LOCK_DIR=/root
  # The precheck: the kernel journal answers with a line, one `[timing-card]` line; else 69 and the reason.
  pc='timing_cards_precheck || { rc=$?; echo "$TWOCARD_WHY"; exit "$rc"; }'
  locx 'timing-card.sh local: the precheck reads the kernel journal' 0 '^\[timing-card\] Xid reader: journalctl -k$' "$pc"
  locx 'timing-card.sh local: the precheck refuses a journal that fails (69)' 69 '^journalctl -k does not answer \(rc 1\): ' "$pc" STUB_JOURNAL=fail
  locx 'timing-card.sh local: the precheck refuses a journal that shows no line (69)' 69 '^journalctl -k shows no kernel line ' "$pc" STUB_JOURNAL=empty
  locx 'timing-card.sh local: the precheck refuses a journal that only warns on stderr (69)' 69 '^journalctl -k shows no kernel line ' \
    "$pc" STUB_JOURNAL=warn
  # After an arm: no Xid and the card answering is 0; the card lost fails the arm by name.
  arm='timing_cards_start > /dev/null; timing_cards_arm log; echo "arm 0"; export STUB_GONE=1; timing_cards_arm log || echo "arm $?: $TWOCARD_WHY"'
  locx 'timing-card.sh local: a lost card fails the arm' 0 "^arm 1: the timing card \\($FAKE_GB10\\) does not answer nvidia-smi \\(rc 15\\)\$" "$arm"
  has 'timing-card.sh local: the card answering passes the arm' '^arm 0$'
  # arm-place.sh: one card, so bp (two cards by size rank) is refused by name, as gate (the card named 3090) is.
  locx 'timing-card.sh local: arm-place.sh refuses bp (64)' 64 \
    "^r: BLOOMERY_GEN_PLACE=bp: local mode times this machine's one card, at a \\(gate loads the card named 3090, bp two cards by size rank\\)\$" \
    'source "${1%/*}/arm-place.sh"; PLACE_RUNNER=r PLACE_BIN=b PLACE_WORDS="a gate bp"; place_check "" bp'
fi

# The t table: the four readers print the same t for df 1..40 (tdump.sh), and it is tdist.py's.
if "$HERE/tdump.sh" "$ROOT" > "$tmp/tdump" 2>&1; then
  bad=$(awk 'NR > 1 && !($2 == $3 && $3 == $4 && $4 == $5) { print }' "$tmp/tdump")
  rows=$(awk 'NR > 1' "$tmp/tdump" | wc -l | tr -d ' ')
  want=$(python3 "$ROOT/tools/ref/tdist.py" 40 | tr ' ' '\n' | awk '{ printf "%d %.4f\n", NR, $1 }')
  got=$(awk 'NR > 1 { print $1, $2 }' "$tmp/tdump")
  if [ -n "$bad" ] || [ "$rows" != 40 ]; then fail 't dump: four readers, one t per df' "$rows rows; disagreeing: $bad" "$tmp/tdump"
  elif [ "$want" != "$got" ]; then fail 't dump: the readers print tdist.py' 'the dump differs from tdist.py' "$tmp/tdump"
  else pass 't dump: gpu-ab, card.py and both depth runners print tdist.py for df 1..40'; fi
else
  fail 't dump runs' "tdump.sh exited $?" "$tmp/tdump"
fi

# gate-batch.sh's walker (--classes) on a tree of its own: a gpu-gate.sh call two hops down in each card
# form, a cycle, a chain past WALK_DEPTH, a lease take two hops down, a script named only in a message,
# and a stub-test harness. It needs just (the Mac; the box has none).
if command -v just > /dev/null; then
  W=$tmp/wtree
  mkdir -p "$W/tools/sub" "$W/tools/x-tests"
  cp "$ROOT/tools/gate-batch.sh" "$W/tools/"
  : > "$W/tools/gate-batch-resources.tsv" # the pack's table, which every plan reads: no solo recipe here, no row
  printf '#!/usr/bin/env bash\necho "usage: gpu-gate.sh — the runner; its own text is not followed"\n' > "$W/tools/gpu-gate.sh"
  printf 'bash tools/sub/b-any.sh\n' > "$W/tools/a-any.sh"
  printf '# a comment naming tools/gpu-gate.sh is no call\nBLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh x\n' > "$W/tools/sub/b-any.sh"
  printf 'S="${BASH_SOURCE[0]%%/*}/sub/b-3090.sh"\nbash "$S"\n' > "$W/tools/a-3090.sh"
  printf 'if ! bash tools/gpu-gate.sh x; then exit 1; fi\n' > "$W/tools/sub/b-3090.sh"
  printf 'bash tools/cyc-2.sh\n' > "$W/tools/cyc-1.sh"
  printf 'bash tools/cyc-1.sh\nBLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh x\n' > "$W/tools/cyc-2.sh"
  for i in 1 2 3 4 5 6; do printf 'bash tools/deep-%d.sh\n' $((i + 1)) > "$W/tools/deep-$i.sh"; done
  printf 'true\n' > "$W/tools/deep-7.sh"
  printf 'bash tools/sub/l2.sh\n' > "$W/tools/l1.sh"
  printf 'source tools/sub/nothing.sh 2> /dev/null || true\nlease_take\n' > "$W/tools/sub/l2.sh"
  printf 'echo "this names tools/sub/b-3090.sh and runs nothing"\n' > "$W/tools/msg.sh"
  printf 'BLOOMERY_GATE_CARD=a6000 bash tools/gpu-gate.sh x\n' > "$W/tools/unread.sh"
  printf 'bash tools/gpu-gate.sh x\n' > "$W/tools/x-tests/harness.sh"
  {
    for r in any:a-any 3090:a-3090 cycle:cyc-1 deep:deep-1 lease:l1 msg:msg unread:unread; do
      printf 'r-%s:\n    ./tools/box.sh '"'"'bash tools/%s.sh'"'"'\n\n' "${r%%:*}" "${r#*:}"
    done
    printf 'r-harness:\n    ./tools/x-tests/harness.sh\n'
  } > "$W/justfile"
  run 'walker: --classes on the stub tree' 0 '^r-any	' env BLOOMERY_GATE_TIMES="$tmp/no-times.tsv" "$W/tools/gate-batch.sh" --classes
  has 'walker: a two-hop any call is balanced with both cards, and names its line' '^r-any	F	balanced	3090,a6000	r-any: gpu-gate\.sh any \(tools/sub/b-any\.sh:2\)$'
  has 'walker: a two-hop call with no card (bare name via BASH_SOURCE) is the 3090' '^r-3090	A	fixed	3090	r-3090: gpu-gate\.sh 3090 \(tools/sub/b-3090\.sh:1\)$'
  has 'walker: a cycle ends, each file read once' '^r-cycle	F	balanced	3090,a6000	r-cycle: gpu-gate\.sh any \(tools/cyc-2\.sh:2\)$'
  has 'walker: a chain past WALK_DEPTH is a named refusal' '^r-deep	R	refused	-	r-deep: the scripts it runs go deeper than WALK_DEPTH = 6 hops \(tools/deep-7\.sh <- tools/deep-6\.sh <- '
  has 'walker: a lease take two hops down is timed' '^r-lease	T	timed	-	r-lease runs tools/sub/l2\.sh <- tools/l1\.sh, which takes the timing lease \(tools/sub/l2\.sh:2 calls lease_take\)'
  has 'walker: a script named in a message is not followed' '^r-msg	F	balanced	-	no device code$'
  has 'walker: a card form the runner does not read is refused' '^r-unread	R	refused	-	r-unread: a script sets BLOOMERY_GATE_CARD in a form this runner does not read: tools/unread\.sh:1'
  has 'walker: a stub-test harness is named, not read' '^r-harness	F	balanced	-	no device code \(r-harness: tools/x-tests/harness\.sh not read \(a stub-test harness\)\)$'
else
  echo "skip walker tests: no just on PATH (check-recipes runs them on the Mac)"
fi

# The cpu guard (guard_cpu, cpu_busy_reading) on a stub /proc tree, and live where /proc exists.
run 'the cpu guard tests (cpu-guard.sh)' 0 '^cpu-guard: [0-9]+ tests, 0 failed$' bash "$HERE/cpu-guard.sh"

# The depth-serve runner's stub tests were started first (below the setup): their result is judged here, with the run() verdict.
wait "$depth_pid"
drc=$?
if [ "$drc" != 0 ]; then
  fail "the depth-serve runner's stub tests (depth-serve.sh --self-test)" "rc $drc, want 0" "$tmp/depth-serve.out"
elif ! grep -Eq -- '^depth-serve\.sh self-test: [0-9]+ checks ok$' "$tmp/depth-serve.out"; then
  fail "the depth-serve runner's stub tests (depth-serve.sh --self-test)" 'no line matches /^depth-serve\.sh self-test: [0-9]+ checks ok$/' "$tmp/depth-serve.out"
else
  pass "the depth-serve runner's stub tests (depth-serve.sh --self-test)"
fi

echo "card-tests: $n tests, $failed failed"
[ "$failed" = 0 ]
