#!/usr/bin/env bash
# batch-replay — a landing batch's schedule, replayed with stub execution on the Mac. Two arms on
# the same stub, each with its own justfile: BASE (git show <BASE>:tools/gate-batch.sh and :justfile)
# and NEW (this tree's two). Every item runs through the real scheduler — the real justfile's
# groups, forms and families, the real lane/chain/steal code — but the box is a fake that sleeps the
# item's measured seconds from a run.log, divided by a scale, and logs `start end item card chain`
# per item. It prints both walls × scale, their difference, the chain's total and its gaps, and
# asserts from its own log that no two chain items overlapped.
#   tools/batch-replay.sh [--scale N] [--base REV] LIST RUNLOG
# LIST is a --list file of items (the raw `just affected` output or one item per line); RUNLOG is
# the measured batch's run.log (`NAME rc=0 Ns try=… lane=…`). What the stub does not model, its
# report says: no builds (the real lanes share one cargo lock), no box.sh guard waits, and sleeps
# in place of work — the page-cache warmth of a family switch (the paper's ①) is outside it.
# A per-item overhead (just's spawn, the batch's bookkeeping) is measured per arm as
# (wall − Σ slept) / items; the difference of the two arms' walls carries both arms' overheads, and
# the arms run the same items, so it cancels to first order.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
SCALE=20 BASE_REV=b3a73bfd OUT=''
while [ $# -gt 2 ]; do
  case "$1" in
    --scale) SCALE=$2; shift 2 ;;
    --base) BASE_REV=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) echo "batch-replay: unknown option '$1'" >&2; exit 64 ;;
  esac
done
[ $# = 2 ] || { echo "usage: tools/batch-replay.sh [--scale N] [--base REV] LIST RUNLOG" >&2; exit 64; }
case $SCALE in '' | *[!0-9]* | 0) echo "batch-replay: --scale is a whole number from 1, got '$SCALE'" >&2; exit 64 ;; esac
LIST=$1 RUNLOG=$2
[ -f "$LIST" ] || { echo "batch-replay: $LIST: not a file" >&2; exit 64; }
[ -f "$RUNLOG" ] || { echo "batch-replay: $RUNLOG: not a file" >&2; exit 64; }
command -v just > /dev/null || { echo "batch-replay: just is not on PATH" >&2; exit 64; }
command -v python3 > /dev/null || { echo "batch-replay: python3 is not on PATH" >&2; exit 64; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/batch-replay.XXXXXX") || exit 70
trap 'rm -rf "$WORK"' EXIT
[ -n "$OUT" ] || OUT=$WORK/out
mkdir -p "$OUT"

# The measured seconds, lane and card per item, and the two arms' trees: BASE from git objects
# (its runner refuses any group outside its own closed set, so it cannot read the new justfile),
# NEW from this tree. ref-paths.sh is the family table the plan reads; the tier is the real one.
python3 - "$RUNLOG" "$WORK/secs.tsv" << 'PY'
import re, sys

rows = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    for ln in fh:
        m = re.match(r"^(\S+) rc=\d+ (\d+)s try=\d+ lane=(\w)", ln)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for name, (secs, lane) in rows.items():
        out.write(f"{name}\t{secs}\t{lane}\n")
print(f"batch-replay: {len(rows)} measured items from {sys.argv[1]}")
PY
for arm in base new; do
  mkdir -p "$WORK/$arm/tools/ref"
  if [ "$arm" = base ]; then
    git -C "$ROOT" show "$BASE_REV:tools/gate-batch.sh" > "$WORK/$arm/tools/gate-batch.sh"
    git -C "$ROOT" show "$BASE_REV:justfile" > "$WORK/$arm/justfile"
  else
    cp "$ROOT/tools/gate-batch.sh" "$WORK/$arm/tools/gate-batch.sh"
    cp "$ROOT/justfile" "$WORK/$arm/justfile"
    # The pack's resource table (a BASE predating it has none to copy; its runner reads none)
    if [ -f "$ROOT/tools/gate-batch-resources.tsv" ]; then
      cp "$ROOT/tools/gate-batch-resources.tsv" "$WORK/$arm/tools/gate-batch-resources.tsv"
    fi
  fi
  cp "$ROOT/tools/ref/ref-paths.sh" "$WORK/$arm/tools/ref/ref-paths.sh"
  # real-only.sh: the plan's real-only validation reads it in the tree (its absence would be a named
  # error for all 44 real-only recipes of the real justfile, in both arms)
  cp "$ROOT/tools/ref/real-only.sh" "$WORK/$arm/tools/ref/real-only.sh"
done

# Each arm's table: the first box.sh command of every listed recipe, exactly as just substitutes
# it (`just --dry-run NAME` prints the body with the parameters in place), mapped to the item's
# measured seconds. A command two items share is a named error: the stub could not tell them apart.
python3 - "$LIST" "$WORK/secs.tsv" "$WORK/base/table.tsv" "$WORK/new/table.tsv" "$WORK/base/justfile" "$WORK/new/justfile" << 'PY'
import re, subprocess, sys

listfile, secs_path, base_t, new_t, base_jf, new_jf = sys.argv[1:7]
with open(listfile, encoding="utf-8") as fh:
    lines = fh.read().split("\n")
head = [ln for ln in lines if ln.strip()][:2]
if head and head[0].startswith("./tools/affected-gates.sh ") and len(head) > 1:
    head = head[1:]
if head and head[0].startswith("affected:"):
    items = [ln.split()[0] for ln in lines if ln.startswith(("  gate-", "  weekly-"))]
else:
    items = [ln.strip().split()[0] for ln in lines
             if ln.strip() and not ln.lstrip().startswith("#") and ln.startswith(("gate-", "weekly-"))]
secs = {}
with open(secs_path, encoding="utf-8") as fh:
    for ln in fh:
        f = ln.rstrip("\n").split("\t")
        secs[f[0]] = f[1]
for table, jf in [(base_t, base_jf), (new_t, new_jf)]:
    rows, seen = [], {}
    for name in items:
        r = subprocess.run(["just", "--justfile", jf, "--dry-run", name],
                           capture_output=True, text=True, check=True)
        dry = r.stdout + r.stderr  # just prints a recipe's dry run to stderr
        cmd = next((ln.strip() for ln in dry.split("\n") if "./tools/box.sh '" in ln), None)
        if cmd is None:
            sys.exit(f"batch-replay: {name}: no box.sh command in its recipe")
        body = cmd.split("./tools/box.sh '", 1)[1].rsplit("'", 1)[0]
        if body in seen:
            sys.exit(f"batch-replay: {name} and {seen[body]} share the box command {body!r}: the stub cannot tell them apart")
        seen[body] = name
        rows.append(f"{body}\t{name}\t{secs.get(name, '45')}")
    with open(table, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
print(f"batch-replay: {len(items)} items per arm; first commands keyed")
PY

# The fake box: the lease probe and the start check pass, a listed command sleeps its seconds and
# logs `start end item card`, anything else exits at once (a later call of a multi-call recipe).
for arm in base new; do
  cat > "$WORK/$arm/tools/box.sh" << FB
#!/bin/sh
here=\$(dirname "\$0")/..
st=\$here/fake-state
mkdir -p "\$st"
case "\$*" in
  '. tools/ref/lease-probe.sh && lease_free')
    echo probe >> "\$st/probe.log"
    exit 0 ;;
  true | 'python3 tools/recipes.py box-manifest') exit 0 ;;
esac
row=\$(awk -F'\\t' -v c="\$*" '\$1 == c { print \$2 "\\t" \$3; exit }' "\$here/table.tsv")
card=-
case " \${BLOOMERY_BOX_ENV:-} " in *BLOOMERY_GATE_CARD=3090*) card=3090 ;; *BLOOMERY_GATE_CARD=a6000*) card=a6000 ;; esac
if [ -z "\$row" ]; then
  printf '%s\\t%s\\t%s\\t%s\\tother\\n' "\$(date +%s)" "\$(date +%s)" "\${BLOOMERY_BOX_ENV:-}" "\$*" >> "\$st/ran.log"
  exit 0
fi
name=\$(printf '%s' "\$row" | cut -f1)
secs=\$(printf '%s' "\$row" | cut -f2)
s0=\$(date +%s); k=0
want=\$((secs / $SCALE + 1))
while [ "\$k" -lt "\$want" ]; do sleep 1; k=\$((k + 1)); done
printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "\$s0" "\$(date +%s)" "\$name" "\$card" "\$*" >> "\$st/ran.log"
exit 0
FB
  chmod +x "$WORK/$arm/tools/box.sh"
  # The plan's expected seconds are the measured ones, so the balance and the chain model reality.
  awk -F'\t' '{ print $1 "\t" ($3 == "A" ? "A" : "B") "\t3090\t" $2 "\t2026-10-09T04:24:49+0900" }' \
    "$WORK/secs.tsv" > "$WORK/$arm/times.tsv"
done

# The two arms, side by side (independent trees and outs; the machine is only sleeping).
run_arm() { # $1 = arm label, $2.. = the batch's own flags
  local arm=$1 rc=0
  shift
  ( cd "$WORK/$arm" && BLOOMERY_GATE_TIMES="$WORK/$arm/times.tsv" \
    bash "$WORK/$arm/tools/gate-batch.sh" --out "$OUT/$arm" --list "$LIST" "$@" ) > "$OUT/$arm.log" 2>&1 || rc=$?
  printf '%s' "$rc" > "$OUT/$arm.rc"
}
# A base with the budget refusal (a32023da and later) needs --over-budget-ok as much as the new arm;
# one without the flag would refuse the unknown option, so it is passed only where it exists.
BASE_BUDGET_OK=''
if git -C "$ROOT" show "$BASE_REV:tools/gate-batch.sh" | grep -q -- --over-budget-ok; then BASE_BUDGET_OK=1; fi
run_arm new --over-budget-ok &
newpid=$!
if [ -n "$BASE_BUDGET_OK" ]; then run_arm base --over-budget-ok; else run_arm base; fi &
basepid=$!
wait "$newpid"; wait "$basepid"
for arm in base new; do
  rc=$(cat "$OUT/$arm.rc")
  [ "$rc" = 0 ] || { echo "batch-replay: the $arm arm ended rc=$rc:" >&2; tail -5 "$OUT/$arm.log" >&2; exit 70; }
done

# Both walls × scale, their difference, the chain's total and gaps, the overlap assertion.
python3 - "$OUT" "$SCALE" "$WORK/new" << 'PY'
import re, subprocess, sys

out, scale, newtree = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def wall(arm):
    with open(f"{out}/{arm}/run.log", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("DONE "):
                m = re.search(r"wall=(\d+)s", ln)
                if m:
                    return int(m.group(1))
    sys.exit(f"batch-replay: no DONE line in {arm}'s run.log")

def measured(arm):
    got = []
    with open(f"{out}/{arm}/run.log", encoding="utf-8") as fh:
        for ln in fh:
            m = re.search(r" rc=\S+ (\d+)s try=", ln)
            if m:
                got.append(int(m.group(1)))
    return got

def chain_names():
    text = subprocess.run(["bash", f"{newtree}/tools/gate-batch.sh", "--classes"],
                          capture_output=True, text=True, check=True).stdout
    return {f[0] for f in (ln.split("\t") for ln in text.split("\n")) if len(f) > 1 and f[1] == "C"}

ran = []
with open(f"{newtree}/fake-state/ran.log", encoding="utf-8") as fh:
    for ln in fh:
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3 and f[2] != "":
            ran.append((int(f[0]), int(f[1]), f[2]))
slept = sum(e - s for s, e, _ in ran)
items = len({name for _, _, name in ran})
chain = chain_names()
iv = sorted((s, e, n) for s, e, n in ran if n in chain)
overlap = [n for (s1, e1, n), (s2, e2, _) in zip(iv, iv[1:]) if s2 < e1]
gaps = sum(max(0, iv[i + 1][0] - iv[i][1]) for i in range(len(iv) - 1))
total = sum(e - s for s, e, _ in iv)
wb, wn = wall("base"), wall("new")
ohb = (wb * scale - sum(measured("base"))) / items
ohn = (wn * scale - sum(measured("new"))) / items
print(f"base: wall {wb}s x{scale} = {wb * scale}s")
print(f"new:  wall {wn}s x{scale} = {wn * scale}s")
print(f"difference: {wn * scale - wb * scale}s (negative is the new schedule's saving)")
print(f"new: {len(iv)} chain items, {total}s x{scale} = {total * scale}s of members, {gaps}s x{scale} = {gaps * scale}s of gaps between them")
print(f"overhead per item: base {ohb:.1f}s, new {ohn:.1f}s (just's spawn and the batch's bookkeeping; the same items in both arms, so the difference carries it to first order)")
if overlap:
    print(f"batch-replay: chain items overlapped: {sorted(set(overlap))}", file=sys.stderr)
    sys.exit(1)
print("chain overlap: none (asserted from the new arm's own log)")
PY
