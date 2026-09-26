#!/usr/bin/env bash
# The card and lease stub tests: tools/ref/card.py on the fixtures here and on docs/cards/, lease_take
# (tools/ref/lease.sh) and its holder naming, tools/ref/lease-hold.sh, tools/ref/card-precheck.sh,
# tools/gpu-ab.py's card check, the witness fields with a failing nvidia-smi, the t table's four
# readers (tdump.sh) and tools/gate-batch.sh's script walker (--classes on a stub tree).
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
cp "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/lease-hold.sh" "$ROOT/tools/ref/card.py" "$ROOT/tools/ref/tdist.py" \
  "$ROOT/tools/ref/card-precheck.sh" "$T/tools/ref/"
cp "$ROOT/tools/gpu-ab.py" "$T/tools/"
cp "$ROOT"/docs/cards/*.card "$HERE/hcpre-ab.card" "$HERE/exclusive.card" "$T/docs/cards/"
echo 'BLOOMERY_BOX_ENV' > "$T/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/old/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/nocard/tools/box.sh"
cp "$ROOT/tools/ref/card.py" "$ROOT/tools/ref/tdist.py" "$tmp/nocard/tools/ref/"

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
    --out "$tmp/gpuab" --recipe time-gpu-v41 "$@"
}

absent() {
  if [ -e "$2" ]; then fail "$1" "$2 exists: the lease file was opened before the refusal"; else pass "$1"; fi
}
free() {
  if flock -n "$2" true; then pass "$1"; else fail "$1" "$2 is still locked"; fi
}
held() {
  if flock -n "$2" true; then fail "$1" "$2 is free"; else pass "$1"; fi
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
card exclusive-long.card 0 'over 30 minutes needs the user.s approval'
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
  'BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab.card BLOOMERY_HOT_LIST=h.txt' \
  --rounds 5 --arm base="$tmp/old" --arm new="$T" --arm lever="$T:BLOOMERY_X=1"
has 'gpu-ab names the tree that checks nothing' "^gpu-ab: $tmp/old predates the card check"
has 'gpu-ab merges BLOOMERY_BOX_ENV' "BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab\\.card BLOOMERY_HOT_LIST=h\\.txt BLOOMERY_AB_ROUNDS=5'"
has 'gpu-ab puts the arm levers after the caller' "BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/hcpre-ab\\.card BLOOMERY_HOT_LIST=h\\.txt BLOOMERY_X=1 BLOOMERY_AB_ROUNDS=5'"
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
    bash -c 'if flock -n "$1" true; then exit 3; fi' _ "$L"
  hold "$T" 'lease-hold marks its command' 0 '^held-by=[0-9]+$' "$L" --card docs/cards/exclusive.card -- \
    bash -c 'echo "held-by=$BLOOMERY_LEASE_HELD"'
  # A process the command leaves running inherits descriptor 9 and keeps the lease until it exits
  # (heavy work beside the next sitting would contaminate it); the next waiter names it, then takes
  # the lease when it ends. The waiter reads the fake /proc tree (the Mac has none): what is tested
  # here is that a busy lease prints its holder at once; the real /proc is the box's run.
  hold "$T" 'lease-hold command leaves a process behind' 0 '^\[lease-hold\] rc=0 ' "$L" --card docs/cards/exclusive.card -- \
    bash -c 'sleep 4 > /dev/null 2>&1 & exit 0'
  has 'lease-hold says the lease outlives it' '^\[lease\] this process let go at .*, but the lease is still held'
  held 'the process left behind keeps the lease' "$L"
  fakeproc "$tmp/proc" "$L"
  take "$T" 'a waiter names the holder at once' 0 "^\\[lease\\] $L is held; waiting up to 30 min\\. Its holder:\$" "$L" \
    BLOOMERY_LEASE_CARD=docs/cards/exclusive.card BLOOMERY_LEASE_PROC="$tmp/proc"
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
  hold "$T" 'lease-hold refuses a card outside docs/cards' 66 'refused CARD_ABSENT' "$tmp/h2.lock" --card tools/ref/card-tests/valid-ab.card -- true
  absent 'lease-hold refused: no lease file opened' "$tmp/h2.lock"
  hold "$ROOT" 'this tree: lease-hold with the self-test card' 0 '^\[lease\] card: ok kind=exclusive minutes=1 ' "$tmp/h3.lock" \
    --card docs/cards/selftest-lease-hold.card -- true
  free 'this tree: the self-test lease is released' "$tmp/h3.lock"
fi

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

echo "card-tests: $n tests, $failed failed"
[ "$failed" = 0 ]
