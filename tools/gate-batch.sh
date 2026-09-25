#!/usr/bin/env bash
# The lead's gate batch runner, two lanes. Runs on the Mac; every item is `just <recipe> [ARGS]`, so
# the box is reached only through the recipes (tools/box.sh) and each item keeps the bound and the
# exit code its recipe's runner owns (tools/gate.sh, tools/gpu-gate.sh, 900 s). This script adds no
# second bound.
#   tools/gate-batch.sh [--out DIR] [--smoke | --list FILE | ITEM…] [--dry-run] [--lanes 1|2]
#
# Items. `NAME[@K=V[,K=V…]][:ARGS]` — NAME a recipe in `just --dump`; `@K=V,…` added to
# BLOOMERY_BOX_ENV for that item (values without spaces, commas or colons); `:ARGS` passed to the
# recipe, word-split (shell quoting allowed). Env comes before ARGS, e.g.
#   gate-gpu-ds41-prefill@BLOOMERY_CARD_EXPERTS=slot:--cases 512 --no-split --no-extra
# --list FILE: one item per line (blank lines and `#` lines skipped), or the raw output of `just
# affected …` (first line `affected:`): then only lines starting with two spaces and `gate-` count, the
# first word is the recipe, everything else is ignored — the `always:` checks are not taken from it.
# --smoke: the fixed smoke list (SMOKE below; docs/gates-plan.md 3.1). An unknown recipe, a malformed
# item, ARGS given to a recipe with no parameters, or an empty list is a named error before anything
# runs.
#
# Lanes (--lanes 2, the default), read from each recipe's text in `just --dump` (and its dependencies'):
#   A  the 3090: a tools/gpu-gate.sh call without BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} (V4.1
#      `--place gate` gates, the V2-Lite p-gates), or device code run with no gate lock at all
#      (`tools/gate.sh --oxide`, `cargo oxide test|run`, a `target/release/` binary of a recipe that
#      runs `cargo oxide`: it lands on the box env's 3090 pin). A gpu-gate.sh recipe here runs with
#      BLOOMERY_GATE_CARD=3090 in BLOOMERY_BOX_ENV, so a mixed recipe's `any` calls stay on the 3090.
#   B  every gpu-gate.sh call carries the `any` form (forced to BLOOMERY_GATE_CARD=a6000, so it never
#      queues on the 3090 lock), plus every recipe that runs no device code (CPU gates, check, lint,
#      fmt-check, check-*, gate-ptx-spill).
#   X  box.sh's own card pick BLOOMERY_CARD=both|a6000 (gate-gpu-load-v41*): the recipe holds both cards
#      without the A6000 gate lock, so it runs alone, after both lanes end.
# A recipe that names a timing runner or the timing lease is refused: a batch's builds contaminate a
# timed run. Each lane runs its items in order; A and B run concurrently (cargo serializes their builds
# by its own lock). --lanes 1 runs every item in the list's order in one lane (labelled A), with no
# card forced — the recipes' own defaults, as a hand batch runs them.
#
# Per item: stdout+stderr to DIR/g-<recipe>[-<n>].log (n for the n-th repeat of a recipe; each try
# appends under a `=== try` header). rc 75 (lock contention) retries after 30 s, up to 10 times; any
# other rc is final. DIR/run.log gets `<recipe>[-<n>] rc=<n> <s>s try=<t> lane=<A|B|X>` per item (plus
# `item=…` when it carries env or ARGS), then `DONE total=<n> red=<n> wall=<s>s laneA=<s>s laneB=<s>s`
# (`laneX=` when X ran, `lint_warnings=<n>` when lint ran: `grep -c '^warning:'` on its log). Exit 0
# iff every rc is 0. DIR defaults to target/gate-batch/<stamp> of this tree: under target/ it is
# gitignored and outside box.sh's rsync, so the logs neither ship to the box nor mark the tree dirty.
# A DIR inside the tree but outside target/ is refused for that reason.
#
# Refuses to start while the timing lease (/root/bloomery-cpu.lock or /root/bloomery-lease.lock) is
# held, naming the lock (exit 75); no override. --dry-run validates, prints the lanes and commands, and
# touches neither the box nor DIR.
#
# Stopping a batch (INT/TERM): the trap signals only the pids written at spawn under DIR (lane-*.pid,
# lane-*.child). A box process a killed item left behind is `just box-gc`'s to clear.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
JUST=(just --justfile "$ROOT/justfile" --working-directory "$ROOT")

SMOKE=(
  check-recipes check-rustflags check-comments check-arch check fmt-check lint gate-ptx-spill
  gate-gpu-ds41-step
  "gate-gpu-ds41-prefill:--cases 512 --no-split --no-extra"
  gate-gpu-e2e
)
TRIES_MAX=11
RETRY_WAIT=30
LEASES="/root/bloomery-cpu.lock /root/bloomery-lease.lock"

USAGE="usage: tools/gate-batch.sh [--out DIR] [--smoke | --list FILE | ITEM…] [--dry-run] [--lanes 1|2]"
die() { echo "gate-batch: $*" >&2; exit "${RC:-64}"; }

OUT='' SRC='' LIST='' DRY=0 LANES=2
ITEMS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || die "--out needs a directory; $USAGE"; OUT=$2; shift 2 ;;
    --smoke) [ -z "$SRC" ] || die "--smoke, --list and items are exclusive; $USAGE"; SRC=smoke; shift ;;
    --list) [ $# -ge 2 ] || die "--list needs a file; $USAGE"
      [ -z "$SRC" ] || die "--smoke, --list and items are exclusive; $USAGE"; SRC=list; LIST=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --lanes) [ $# -ge 2 ] || die "--lanes needs 1 or 2; $USAGE"
      case "$2" in 1 | 2) LANES=$2 ;; *) die "--lanes is 1 or 2, got '$2'" ;; esac; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    --*) die "unknown option '$1'; $USAGE" ;;
    *) [ -z "$SRC" ] || [ "$SRC" = items ] || die "--smoke, --list and items are exclusive; $USAGE"
      SRC=items; ITEMS+=("$1"); shift ;;
  esac
done
[ -n "$SRC" ] || die "no items; $USAGE"
if [ "$SRC" = list ]; then
  [ -f "$LIST" ] && [ -r "$LIST" ] || die "--list $LIST: not a readable file"
fi
[ "$SRC" = smoke ] && ITEMS=("${SMOKE[@]}")
command -v just > /dev/null || die "just is not on PATH"
command -v python3 > /dev/null || die "python3 is not on PATH"

# The plan: one record per item, fields split by \x1f — lane, log stem, recipe, env for
# BLOOMERY_BOX_ENV (space separated), ARGS (shell-quoted for eval), the item as written, the reason.
# (The source sits in a variable: bash 3.2 misparses a heredoc inside $(…) that holds quotes.)
IFS= read -r -d '' PYPLAN << 'PY' || true
import json, re, shlex, subprocess, sys

root, lanes, src, listfile, caller_env = sys.argv[1:6]
raw_items = sys.argv[6:]
just = ["just", "--justfile", root + "/justfile", "--working-directory", root]


def fail(msg):
    print("gate-batch: " + msg, file=sys.stderr)
    sys.exit(65)


if src == "list":
    with open(listfile, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if first.startswith("affected:"):
        raw_items = [ln.split()[0] for ln in lines if ln.startswith("  gate-")]
        where = [f"{listfile} (just affected output)"] * len(raw_items)
    else:
        raw_items, where = [], []
        for i, ln in enumerate(lines, 1):
            if ln.strip() and not ln.lstrip().startswith("#"):
                raw_items.append(ln.strip())
                where.append(f"{listfile}:{i}")
else:
    where = [src] * len(raw_items)
if not raw_items:
    fail(f"the {src} input names no item — nothing to run is not a green batch")

proc = subprocess.run(just + ["--dump", "--dump-format", "json"], capture_output=True, text=True)
if proc.returncode != 0:
    fail(f"`just --dump` failed (exit {proc.returncode}): {proc.stderr.strip()}")
recipes = json.loads(proc.stdout)["recipes"]


def body(name):
    out = []
    for frags in recipes[name]["body"]:
        out.append("".join(f if isinstance(f, str) else "{{}}" for f in frags))
    return "\n".join(out)


def closure(name, seen):
    if name in seen:
        return []
    seen.add(name)
    names = [name]
    for dep in recipes[name]["dependencies"]:
        names += closure(dep["recipe"], seen)
    return names


GPU_GATE = "tools/gpu-gate.sh"
ANY = "BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}"
TIMED = re.compile(
    r"tools/ref/(time-gate|timing-card|measure|cpu-measure|depth-[a-z0-9]+|nsys-gpu|ncu-gpu)\.sh"
    r"|gpu-ab\.py|/root/bloomery-(cpu|lease)\.lock"
)
BOX_CARD = re.compile(r"\bBLOOMERY_CARD=(both|a6000)\b")
UNLOCKED = re.compile(r"tools/gate\.sh --oxide|cargo oxide (test|run)\b")
SEGMENT = re.compile(r"&&|\|\||;|\||\n")


def classify(name):
    """(lane, reason) for a recipe and its dependencies; None for lane means refused."""
    lanes_seen, reasons = set(), []
    for n in closure(name, set()):
        text = body(n)
        m = TIMED.search(text)
        if m:
            return None, f"{n} names {m.group(0)}: a timed recipe does not run in a gate batch"
        m = BOX_CARD.search(text)
        if m:
            lanes_seen.add("X")
            reasons.append(f"{n}: {m.group(0)} (both cards, no A6000 gate lock)")
            continue
        oxide = "cargo oxide" in text
        for seg in SEGMENT.split(text):
            if GPU_GATE in seg:
                if ANY in seg:
                    lanes_seen.add("B")
                    reasons.append(f"{n}: gpu-gate.sh any")
                elif "BLOOMERY_GATE_CARD" in seg:
                    return None, f"{n}: a gpu-gate.sh call sets BLOOMERY_GATE_CARD in a form this runner does not read: {seg.strip()}"
                else:
                    lanes_seen.add("A")
                    reasons.append(f"{n}: gpu-gate.sh 3090")
            elif UNLOCKED.search(seg):
                lanes_seen.add("A")
                reasons.append(f"{n}: {UNLOCKED.search(seg).group(0)} (device code, no gate lock)")
            elif oxide and "target/release/" in seg and "host-gate.sh" not in seg:
                lanes_seen.add("A")
                reasons.append(f"{n}: direct target/release run (box env 3090 pin, no gate lock)")
    for lane in ("X", "A", "B"):
        if lane in lanes_seen:
            return lane, "; ".join(dict.fromkeys(r for r in reasons))
    return "B", "no device code"


ITEM = re.compile(r"^([A-Za-z0-9_-]+)(?:@([^:]*))?(?::(.*))?$")
VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
for kv in caller_env.split():
    if lanes == "2" and kv.split("=", 1)[0] == "BLOOMERY_GATE_CARD":
        fail("BLOOMERY_BOX_ENV sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")

counts, errors, out = {}, [], []
for item, at in zip(raw_items, where):
    m = ITEM.match(item)
    if not m:
        errors.append(f"{at}: malformed item {item!r} (NAME[@K=V,…][:ARGS])")
        continue
    name, env, args = m.group(1), m.group(2), m.group(3)
    if name not in recipes:
        errors.append(f"{at}: {name!r} is not a recipe in the justfile")
        continue
    envs = []
    if env is not None:
        for kv in env.split(","):
            k, eq, v = kv.partition("=")
            if not eq or not VAR.match(k) or re.search(r"\s", v):
                errors.append(f"{at}: env entry {kv!r} of {item!r} is not K=V (no spaces)")
            elif lanes == "2" and k == "BLOOMERY_GATE_CARD":
                errors.append(f"{at}: {item!r} sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")
            else:
                envs.append(f"{k}={v}")
    argv = []
    if args is not None:
        if re.search(r"@[A-Za-z_][A-Za-z0-9_]*=", args):
            errors.append(f"{at}: {item!r}: env goes before ARGS (NAME@K=V:ARGS)")
            continue
        try:
            argv = shlex.split(args)
        except ValueError as e:
            errors.append(f"{at}: ARGS of {item!r}: {e}")
            continue
        if argv and not recipes[name]["parameters"]:
            errors.append(f"{at}: {name} takes no arguments (just would read {argv[0]!r} as another recipe)")
            continue
    # just's own argument check (arity, options), printing the recipe instead of running it.
    chk = subprocess.run(just + ["--dry-run", name] + argv, capture_output=True, text=True)
    if chk.returncode != 0:
        msg = [ln for ln in chk.stderr.splitlines() if ln.startswith("error")] or chk.stderr.splitlines()[-1:]
        errors.append(f"{at}: just rejects {item!r}: {' '.join(msg)}")
        continue
    lane, reason = classify(name)
    if lane is None:
        errors.append(f"{at}: {reason}")
        continue
    uses_gpu_gate = any(GPU_GATE in body(n) for n in closure(name, set()))
    if lanes == "1":
        lane = "A"
    elif uses_gpu_gate and lane in ("A", "B"):
        envs.append("BLOOMERY_GATE_CARD=" + ("3090" if lane == "A" else "a6000"))
    counts[name] = counts.get(name, 0) + 1
    stem = name if counts[name] == 1 else f"{name}-{counts[name]}"
    shown = item if (env is not None or args is not None) else ""
    out.append("\x1f".join([lane, stem, name, " ".join(envs), " ".join(shlex.quote(a) for a in argv), shown, reason]))
if errors:
    for e in errors:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(errors)} item error(s); nothing ran")
print("\n".join(out))
PY
PLAN=$(python3 -c "$PYPLAN" "$ROOT" "$LANES" "$SRC" "$LIST" "${BLOOMERY_BOX_ENV:-}" ${ITEMS[@]+"${ITEMS[@]}"}) \
  || RC=$? die "the list did not validate (above)"

P_LANE=() P_STEM=() P_NAME=() P_ENV=() P_ARGS=() P_ITEM=() P_WHY=()
while IFS=$'\x1f' read -r lane stem name env args item why; do
  P_LANE+=("$lane"); P_STEM+=("$stem"); P_NAME+=("$name"); P_ENV+=("$env")
  P_ARGS+=("$args"); P_ITEM+=("$item"); P_WHY+=("$why")
done <<< "$PLAN"
N=${#P_NAME[@]}

box_env_of() { # the item's full BLOOMERY_BOX_ENV: the caller's, then the item's and the lane's
  local e="${BLOOMERY_BOX_ENV:-}"
  [ -z "${P_ENV[$1]}" ] || e="${e:+$e }${P_ENV[$1]}"
  printf '%s' "$e"
}
cmd_of() {
  local e; e=$(box_env_of "$1")
  printf '%sjust %s%s' "${e:+BLOOMERY_BOX_ENV='$e' }" "${P_NAME[$1]}" "${P_ARGS[$1]:+ ${P_ARGS[$1]}}"
}

if [ -z "$OUT" ]; then
  OUT="$ROOT/target/gate-batch/$(date +%Y%m%d-%H%M%S)"
fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$OUT/" in
  "$ROOT"/target/*) ;;
  "$ROOT"/*) die "--out $OUT is inside the tree but not under target/: box.sh would ship the logs and mark the tree dirty" ;;
esac

if [ "$DRY" = 1 ]; then
  echo "gate-batch: dry run — $N items, lanes $LANES, logs would go to $OUT"
  for ((i = 0; i < N; i++)); do
    printf 'lane %s  %-28s %s\n        %s\n' "${P_LANE[$i]}" "${P_STEM[$i]}" "$(cmd_of "$i")" "${P_WHY[$i]}"
  done
  exit 0
fi

mkdir -p "$OUT" || RC=73 die "cannot create the log directory $OUT"
[ -d "$OUT" ] && [ -w "$OUT" ] || RC=73 die "the log directory $OUT is not a writable directory"
[ ! -e "$OUT/run.log" ] || RC=73 die "$OUT already holds a batch (run.log) — give --out a new directory"
RUNLOG=$OUT/run.log

# The timing lease, each lock tested on its own so the refusal names it. flock -E 75 separates "held"
# from "could not test".
lease=$("$ROOT/tools/box.sh" 'for l in '"$LEASES"'; do flock -n -E 75 "$l" true; r=$?; if [ "$r" = 0 ]; then echo "free $l"; elif [ "$r" = 75 ]; then echo "held $l"; else echo "error $l rc=$r"; fi; done' 2>&1) \
  || RC=70 die "the lease check through box.sh failed: $lease"
for l in $LEASES; do
  case "$lease" in
    *"held $l"*) RC=75 die "the timing lease $l is held — a batch's builds contaminate a sitting; not starting" ;;
    *"free $l"*) ;;
    *) RC=70 die "the lease $l could not be tested: $lease" ;;
  esac
done

# run.log exists from here on (a refused start leaves none, so the same --out can be retried).
: > "$RUNLOG"
T0=$(date +%s)
echo "gate-batch: $N items, lanes $LANES, logs in $OUT"

run_item() { # $1 = plan index, $2 = lane label; the lane's current child pid goes to lane-<lane>.child
  local i=$1 lane=$2 log="$OUT/g-${P_STEM[$1]}.log" try=0 rc t0 s benv line
  local argv=()
  eval "argv=(${P_ARGS[$i]})"
  benv=$(box_env_of "$i")
  t0=$(date +%s)
  while :; do
    try=$((try + 1))
    echo "=== try $try $(date '+%F %T') lane=$lane: $(cmd_of "$i")" >> "$log"
    rc=0
    BLOOMERY_BOX_ENV="$benv" "${JUST[@]}" "${P_NAME[$i]}" ${argv[@]+"${argv[@]}"} >> "$log" 2>&1 < /dev/null &
    echo $! > "$OUT/lane-$lane.child"
    wait $! || rc=$?
    if [ "$rc" -eq 75 ] && [ "$try" -lt "$TRIES_MAX" ]; then
      echo "=== rc 75 (lock contention): retry in ${RETRY_WAIT} s" >> "$log"
      sleep "$RETRY_WAIT"
      continue
    fi
    break
  done
  s=$(($(date +%s) - t0))
  line="${P_STEM[$i]} rc=$rc ${s}s try=$try lane=$lane"
  [ -z "${P_ITEM[$i]}" ] || line="$line item=${P_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
}

run_lane() { # $1 = lane label; runs its items in plan order, then writes lane-<lane>.s
  local lane=$1 t0 i
  t0=$(date +%s)
  for ((i = 0; i < N; i++)); do
    [ "${P_LANE[$i]}" = "$lane" ] && run_item "$i" "$lane"
  done
  echo $(($(date +%s) - t0)) > "$OUT/lane-$lane.s"
}

stop() {
  local f
  # The lanes first, so none starts its next item, then the items they were running.
  for f in "$OUT"/lane-*.pid "$OUT"/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  echo "gate-batch: stopped; a box process a killed item left behind is cleared by 'just box-gc'" >&2
  exit 130
}
trap stop INT TERM

has_lane() { local i; for ((i = 0; i < N; i++)); do [ "${P_LANE[$i]}" = "$1" ] && return 0; done; return 1; }

LANES_RUN=""
for lane in A B; do
  if has_lane "$lane"; then
    run_lane "$lane" &
    echo $! > "$OUT/lane-$lane.pid"
    LANES_RUN="$LANES_RUN $lane"
  fi
done
for lane in $LANES_RUN; do
  wait "$(cat "$OUT/lane-$lane.pid")" || true
done
# Every lane is waited for before any is judged: a lane that died must not leave the other running.
for lane in $LANES_RUN; do
  [ -f "$OUT/lane-$lane.s" ] || RC=70 die "lane $lane ended without its sentinel ($OUT/lane-$lane.s) — see run.log"
done
if has_lane X; then run_lane X; fi
trap - INT TERM

recorded=$(grep -c ' rc=' "$RUNLOG" || true)
[ "$recorded" -eq "$N" ] || RC=70 die "$N items planned, $recorded recorded in $RUNLOG"
green=$(grep -c ' rc=0 ' "$RUNLOG" || true)
red=$((N - green))
lane_s() { if [ -f "$OUT/lane-$1.s" ]; then cat "$OUT/lane-$1.s"; else echo 0; fi; }
done_line="DONE total=$N red=$red wall=$(($(date +%s) - T0))s laneA=$(lane_s A)s laneB=$(lane_s B)s"
if has_lane X; then done_line="$done_line laneX=$(lane_s X)s"; fi
for ((i = 0; i < N; i++)); do
  if [ "${P_NAME[$i]}" = lint ]; then
    done_line="$done_line lint_warnings=$(grep -c '^warning:' "$OUT/g-${P_STEM[$i]}.log" || true)"
    break
  fi
done
echo "$done_line" >> "$RUNLOG"
echo "$done_line"
if [ "$red" -ne 0 ]; then
  grep -v ' rc=0 ' "$RUNLOG" | grep ' rc=' | while read -r stem _; do echo "red: $stem  $OUT/g-$stem.log"; done
  exit 1
fi
