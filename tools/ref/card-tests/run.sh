#!/usr/bin/env bash
# The card and lease stub tests: tools/ref/card.py on the fixtures here and on docs/cards/, lease_take
# (tools/ref/lease.sh), tools/ref/lease-hold.sh and tools/gpu-ab.py's card check.
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

# The copy: the lease code and gpu-ab.py, the tree's cards, and fixtures as the run cards
# docs/cards/hcpre-ab.card (ab, rounds 5) and docs/cards/exclusive.card. tools/box.sh only has to
# exist and name BLOOMERY_BOX_ENV (gpu-ab.py reads it); nothing runs it.
T=$tmp/tree
mkdir -p "$T/tools/ref" "$T/docs/cards" "$tmp/old/tools" "$tmp/nocard/tools/ref"
cp "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/lease-hold.sh" "$ROOT/tools/ref/card.py" "$T/tools/ref/"
cp "$ROOT/tools/gpu-ab.py" "$T/tools/"
cp "$ROOT"/docs/cards/*.card "$HERE/hcpre-ab.card" "$HERE/exclusive.card" "$T/docs/cards/"
echo 'BLOOMERY_BOX_ENV' > "$T/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/old/tools/box.sh"
cp "$T/tools/box.sh" "$tmp/nocard/tools/box.sh"
cp "$ROOT/tools/ref/card.py" "$tmp/nocard/tools/ref/"

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
  hold "$T" 'lease-hold command leaves a process behind' 0 '^\[lease-hold\] rc=0 ' "$L" --card docs/cards/exclusive.card -- \
    bash -c 'sleep 3 > /dev/null 2>&1 & exit 0'
  free 'the process left behind does not keep the lease' "$L"
  hold "$T" 'lease-hold refuses a card outside docs/cards' 66 'refused CARD_ABSENT' "$tmp/h2.lock" --card tools/ref/card-tests/valid-ab.card -- true
  absent 'lease-hold refused: no lease file opened' "$tmp/h2.lock"
  hold "$ROOT" 'this tree: lease-hold with the self-test card' 0 '^\[lease\] card: ok kind=exclusive minutes=1 ' "$tmp/h3.lock" \
    --card docs/cards/selftest-lease-hold.card -- true
  free 'this tree: the self-test lease is released' "$tmp/h3.lock"
fi

echo "card-tests: $n tests, $failed failed"
[ "$failed" = 0 ]
