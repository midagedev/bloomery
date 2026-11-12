#!/usr/bin/env bash
# The night's real tier: one command, no machine facts. It runs on the box itself at 00:00 (a timer the machine's own configs
# install), when the box is free, so the day's landings run only the fixture tier and a release runs only what the night did not
# already prove at the tag's keys (tools/rotation.py release-check, tools/release/publish.sh step 0b).
#
#   tools/nightly-real.sh [--commit REV] [--dry-run]   the night: the commit (default origin/main, after `git fetch origin`), its
#                                                     list, the batch, the report; --dry-run stops after the plan
#   tools/nightly-real.sh --pull [--into DIR]           the other side's read of the night (the Mac's, before a release): the box's
#                                                     report, status and ledger into DIR, through tools/box.sh read-only
#   tools/nightly-real.sh --self-test                   stub tools in a temporary repository: the skip, a red with its range, the
#                                                     refusal under a hold, the carried red, the lock, the outer phase, --pull
#
# The night, in order:
#   1. the lock: one night at a time ($NIGHT_HOME/lock, a directory with the holder's pid; one whose pid is gone is taken over);
#   2. the commit: `git fetch origin` and origin/main unless --commit names one; the night's own worktree ($NIGHT_HOME/tree) is
#      moved to it (checkout --detach --force, clean), and that tree's own copy of this script runs the rest, so the list and the
#      batch are the commit's;
#   3. the refusal: one `BLOOMERY_BOX_WAIT=0 tools/box.sh true` — box.sh's own guard (tools/ref/lease-probe.sh) ends it 75 while
#      the timing lease or a hold is up, and the night ends 75 with status `refused`, its previous report kept. tools/gate-batch.sh
#      refuses the same way at its start (its box-manifest read), so a lease that comes up between the two refuses there too;
#   4. the list: `tools/recipes.py release-list <newest v* tag>` in full (the real tier, the blocking set, affected since the tag,
#      the CUDA GGUF gates) and `tools/rotation.py due` on the night's ledger, one recipe a line, sorted, no duplicate. The
#      classifier runs once for both (`--classes`);
#   5. the skip: one `tools/gate-batch.sh --ledger --dry-run --tier real` over the list; when every item is green at its key the
#      night ends 0 with one line and a `skip` report (and no batch);
#   6. the batch: `tools/gate-batch.sh --tier real --ledger --over-budget-ok --list` under BLOOMERY_NIGHT_BOUND seconds
#      (tools/gate-bound.sh bounded_bg). The night runs every item: a gate past 0.75x its bound is not refused (the
#      dry run's budget line goes in the report), it runs to its own bound and a timeout is a red the morning reads;
#   7. the report, $NIGHT_HOME/report.tsv (atomic; each night's copy and its lists stay in $NIGHT_HOME/runs/<stamp>/), tab-separated:
#        night  commit=<sha12> date=<start> state=<ran|skip> rc=<n> items=<n> torun=<n> plan=<s> tag=<tag> ledger=<path> run=<dir>
#        red    <item>  the batch's own line for it (rc, seconds, last_green=<commit>@<date>, range=<a>..<b> commits=<n> touching=…);
#               a red of an earlier night that this one gave no verdict (it left the list, or ended 75) is carried, `carried-from=<date>`
#        unrun  <item>  a final rc 75 or a yield: nothing proved, the last verdict stands
#        done   the batch's DONE line (totals, wall), or `skip …`
#   8. the hook: BLOOMERY_NIGHT_NOTIFY, when set, is a command called as `<command> <state> <file>` (state green, red, skip,
#      refused, failed or stopped; file the report, or status.txt when no report was written). Unset, nothing is told. A hook
#      that fails is one stderr line and never changes the exit code.
#
# Where it runs and where its ledger lives. tools/box.sh has a box-local mode (BLOOMERY_BOX_LOCAL=1) that it refuses on the box by
# design, so the night drives tools/gate-batch.sh on the box against the box itself: box.sh's ssh to its own host name (the box's
# ssh config names itself) and rsync into the night's own remote directory, BLOOMERY_REMOTE (default ~/repo/bloomery-night; two
# trees never share one). The batch's --ledger is the night's own file, $NIGHT_HOME/gate-ledger.tsv, on the box: the lead's
# ledger stays the Mac's. The Mac reads the night with --pull, and tools/rotation.py release-check and check --night read the
# pulled ledger beside its own: a key is the same on both sides when the environments the key reads match
# (tools/recipes.py key's mac-env and mac-tool parts); where they do not, the release runs the blocking set itself.
#
# Environment (all optional): BLOOMERY_NIGHT_HOME (default ~/.cache/bloomery/night), BLOOMERY_REMOTE (default
# ~/repo/bloomery-night), BLOOMERY_NIGHT_BOUND (the batch's seconds, default 21600), BLOOMERY_NIGHT_NOTIFY (above);
# --pull: BLOOMERY_NIGHT_PULL (the night's directory on the box, default ~/.cache/bloomery/night) and --into (default
# BLOOMERY_NIGHT_DIR, else ~/.cache/bloomery/night-pull).
# Exit codes: 0 green or skip; 1 red; 64 usage; 66 a commit, tag or file that is not there; 70 a tool that failed (named);
# 75 refused (the lease or a hold is up, or another night holds the lock); 143 stopped by a signal.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
USAGE="usage: tools/nightly-real.sh [--commit REV] [--dry-run] | --pull [--into DIR] | --self-test"
NIGHT_HOME=${BLOOMERY_NIGHT_HOME:-$HOME/.cache/bloomery/night}
NIGHT_BOUND=${BLOOMERY_NIGHT_BOUND:-21600}
export BLOOMERY_REMOTE=${BLOOMERY_REMOTE:-'~/repo/bloomery-night'}

say() { echo "nightly-real: $*" >&2; }
stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# The report's reader and writer, one Python owner for the batch's lines.
IFS= read -r -d '' PYNIGHT << 'PY' || true
import os, re, sys

LANE = re.compile(r"^lane (\S+)(?: \(simulated\))?\s+(\S+)(?:\s|$)")
LEDGER = re.compile(r"^\s+ledger: (\S+) — ")
WALL = re.compile(r"^gate-batch: predicted .*\bwall=(\d+)s", re.M)
ITEM = re.compile(r"^(\S+) rc=(\S+)")
NOT_RUN = ("75", "yield")


def items(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def plan(path):
    """`<items> <to run> <wall>` of a `gate-batch.sh --ledger --dry-run`: an item under `lane -` is deferred and counts in
    neither; an item with no `ledger: skip` line is one to run."""
    text = open(path, encoding="utf-8").read()
    lanes, st, cur = [], {}, None
    for ln in text.splitlines():
        m = LANE.match(ln)
        if m:
            cur = m.group(2) if m.group(1) != "-" else None
            if cur is not None:
                lanes.append(cur)
            continue
        m = LEDGER.match(ln)
        if m and cur is not None:
            st[cur] = m.group(1)
    w = WALL.search(text)
    if w is None:
        sys.exit(f"{path}: no `gate-batch: predicted … wall=Ns` line")
    print(len(lanes), sum(1 for n in lanes if st.get(n) != "skip"), w.group(1))


def report(runlog, listf, prev, out, *head):
    """The night's report (the header of tools/nightly-real.sh); `-` for the run log is a skip."""
    h = dict(x.split("=", 1) for x in head)
    final, done = {}, ""
    if runlog != "-":
        for ln in open(runlog, encoding="utf-8").read().splitlines():
            if ln.startswith("DONE "):
                done = ln
                continue
            if ln.startswith(("plan ", "ledger ")):
                continue
            m = ITEM.match(ln)
            if not m:
                continue
            it = re.search(r"\bitem=(\S+)$", ln)
            final[it.group(1) if it else m.group(1)] = (m.group(2), ln[len(m.group(1)) + 1:])
    else:
        done = f"skip: every one of the {h.get('items', '?')} item(s) is green at its key"
    listed = set(items(listf))
    reds = {k: v for k, (rc, v) in final.items() if rc not in ("0", "skip", "deferred") + NOT_RUN}
    unrun = {k: v for k, (rc, v) in final.items() if rc in NOT_RUN}
    verdict = {k for k, (rc, _) in final.items() if rc not in ("deferred",) + NOT_RUN}
    carried = {}
    if os.path.exists(prev):
        pdate = ""
        for ln in open(prev, encoding="utf-8").read().splitlines():
            f = ln.split("\t")
            if f[0] == "night":
                pdate = dict(x.split("=", 1) for x in f[1:] if "=" in x).get("date", "")
            elif f[0] == "red" and len(f) >= 3:
                seen = f[1] in listed if runlog == "-" else f[1] in verdict
                if not seen and f[1] not in reds:
                    detail = "\t".join(f[2:])
                    carried[f[1]] = detail if "carried-from=" in detail else f"{detail} carried-from={pdate}"
    lines = [f"# night {h.get('date')} at {h.get('commit')}: {h.get('state')}, {len(reds) + len(carried)} red ({len(carried)} carried), "
             f"{len(unrun)} unrun, {h.get('items', '?')} item(s), {h.get('torun', '?')} to run; {done}",
             "night\t" + "\t".join(f"{k}={v}" for k, v in h.items())]
    lines += [f"red\t{k}\t{v}" for k, v in sorted(reds.items())]
    lines += [f"red\t{k}\t{v}" for k, v in sorted(carried.items())]
    lines += [f"unrun\t{k}\t{v}" for k, v in sorted(unrun.items())]
    lines.append(f"done\t{done}")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, out)
    print(len(reds) + len(carried), len(carried), len(unrun), done)


cmd, *rest = sys.argv[1:]
{"plan": plan, "report": report}[cmd](*rest)
PY

# lock_take DIR: one night at a time. 0 taken; 75 another live night holds it (named). A lock whose holder's pid is gone is
# taken over, named.
lock_take() {
  local lock=$1 pid
  mkdir -p "$(dirname "$lock")"
  if ! mkdir_lock "$lock"; then
    pid=$(cat "$lock/pid" 2> /dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then
      say "refused: another night (pid $pid) holds $lock"
      return 75
    fi
    say "the lock $lock names pid ${pid:-none}, which is gone: taken over"
    rm -rf "$lock"
    mkdir_lock "$lock" || { say "refused: $lock was taken by another night meanwhile"; return 75; }
  fi
  echo $$ > "$lock/pid"
}
lock_drop() { [ -z "${LOCK:-}" ] || rm -rf "$LOCK"; }

# status STATE RC WHY: the one-line status of the night, written whatever it ends in; then the hook.
status() {
  local file=$NIGHT_HOME/status.txt
  printf '%s %s rc=%s %s\n' "$(stamp)" "$1" "$2" "$3" > "$file.tmp" && mv "$file.tmp" "$file"
  notify "$1" "${4:-$file}"
}
notify() {
  [ -n "${BLOOMERY_NIGHT_NOTIFY:-}" ] || return 0
  local rc=0
  # shellcheck disable=SC2086 # the hook is a command and its words
  $BLOOMERY_NIGHT_NOTIFY "$1" "$2" < /dev/null || rc=$?
  [ "$rc" = 0 ] || say "the hook BLOOMERY_NIGHT_NOTIFY ended rc $rc (the night's verdict stands)"
}
# end STATE RC WHY: the status, the hook, the exit.
end() {
  status "$1" "$2" "$3"
  say "$1: $3"
  exit "$2"
}

# ---- --pull: the box's night, read here (tools/box.sh read-only: no sync, no guard wait)
pull() {
  local into=$1 box_dir=${BLOOMERY_NIGHT_PULL:-'~/.cache/bloomery/night'} f rc
  mkdir -p "$into"
  for f in report.tsv status.txt gate-ledger.tsv; do
    rc=0
    BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" "cat $box_dir/$f" < /dev/null > "$into/$f.tmp" 2> "$into/$f.err" || rc=$?
    if [ "$rc" = 0 ]; then
      mv "$into/$f.tmp" "$into/$f"
    elif [ "$f" = report.tsv ]; then
      say "--pull: the box has no night report at $box_dir/$f (box.sh rc $rc: $(tail -1 "$into/$f.err" 2> /dev/null))"
      rm -f "$into/$f.tmp"
      return 66
    else
      rm -f "$into/$f.tmp"
      [ "$f" != gate-ledger.tsv ] || : > "$into/$f"
    fi
    rm -f "$into/$f.err"
  done
  say "--pull: $into/report.tsv: $(head -1 "$into/report.tsv")"
}

# ---- the outer phase: the lock, the commit, the night's own tree, then that tree's copy of this script
outer() {
  local rev=$1 dry=$2 commit tree=$NIGHT_HOME/tree rc=0
  # shellcheck source=tools/nightly/units.sh
  . "$ROOT/tools/nightly/units.sh"
  LOCK=$NIGHT_HOME/lock
  lock_take "$LOCK" || exit $?
  trap lock_drop EXIT
  if [ -z "$rev" ]; then
    git -C "$ROOT" fetch --quiet origin || rc=$?
    [ "$rc" = 0 ] || end failed 70 "git fetch origin ended rc $rc: the night runs the commit origin/main names after a fetch, never an older one"
    rev=origin/main
  fi
  commit=$(git -C "$ROOT" rev-parse --verify --quiet "$rev^{commit}") || end failed 66 "no commit $rev in $ROOT"
  if [ -e "$tree/.git" ]; then
    if ! git -C "$tree" checkout --quiet --detach --force "$commit" || ! git -C "$tree" clean -fdq; then
      end failed 70 "the night's tree $tree did not move to $commit"
    fi
  elif [ -e "$tree" ]; then
    end failed 70 "$tree is there and is no worktree: move it away, the night makes its own"
  else
    git -C "$ROOT" worktree add --quiet --detach "$tree" "$commit" || end failed 70 "git worktree add $tree $commit failed"
  fi
  [ -f "$tree/tools/nightly-real.sh" ] || end failed 66 "$commit has no tools/nightly-real.sh"
  trap - EXIT
  exec env BLOOMERY_NIGHT_LOCK="$LOCK" bash "$tree/tools/nightly-real.sh" --in-tree --commit "$commit" ${dry:+"$dry"}
}

# ---- the inner phase, in the night's tree at its commit
inner() {
  local commit=$1 dry=$2 run rc tag ledger=$NIGHT_HOME/gate-ledger.tsv n_items n_run plan_s red carried unrun done_line state hdr
  # shellcheck source=tools/nightly/units.sh
  . "$ROOT/tools/nightly/units.sh"
  # shellcheck source=tools/gate-bound.sh
  . "$ROOT/tools/gate-bound.sh"
  LOCK=${BLOOMERY_NIGHT_LOCK:-}
  trap 'bounded_bg_stop_all || true; lock_drop' EXIT
  cd "$ROOT"
  run=$NIGHT_HOME/runs/$(date '+%Y%m%dT%H%M%S')
  mkdir -p "$run"
  local start
  start=$(stamp)
  # 3. the refusal: box.sh's own guard, waiting for nothing
  rc=0
  BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" true < /dev/null > "$run/probe.log" 2>&1 || rc=$?
  case $rc in
    0) ;;
    75) end refused 75 "the box's guard refused (the timing lease or a hold is up): $(grep -v '^$' "$run/probe.log" | tail -1); the previous report stands" ;;
    *) end failed 70 "tools/box.sh true ended rc $rc: $(tail -1 "$run/probe.log")" ;;
  esac
  # 4. the list
  "$ROOT/tools/gate-batch.sh" --tier fixture --classes > "$run/classes.tsv" 2> "$run/classes.err" ||
    end failed 70 "tools/gate-batch.sh --classes failed: $(tail -1 "$run/classes.err")"
  tag=$(git -C "$ROOT" describe --tags --abbrev=0 --match 'v*' HEAD 2> /dev/null) || end failed 66 "no v* tag before $commit: the release list is told against the last release"
  python3 tools/recipes.py release-list "$tag" --classes "$run/classes.tsv" > "$run/release.list" 2> "$run/release.err" ||
    end failed 70 "tools/recipes.py release-list $tag failed: $(tail -1 "$run/release.err")"
  python3 tools/rotation.py due --classes "$run/classes.tsv" --ledger "$ledger" --out "$run/due.list" 2> "$run/due.err" ||
    end failed 70 "tools/rotation.py due failed: $(tail -1 "$run/due.err")"
  grep -hv '^[[:space:]]*#' "$run/release.list" "$run/due.list" | sed '/^[[:space:]]*$/d' | sort -u > "$run/night.list"
  # 5. the skip: the batch's own key check
  rc=0
  BLOOMERY_GATE_LEDGER=$ledger "$ROOT/tools/gate-batch.sh" --ledger --dry-run --tier real --over-budget-ok --list "$run/night.list" \
    < /dev/null > "$run/plan.txt" 2>&1 || rc=$?
  [ "$rc" = 0 ] || end failed 70 "the batch's dry run ended rc $rc: $(tail -1 "$run/plan.txt")"
  read -r n_items n_run plan_s < <(python3 -c "$PYNIGHT" plan "$run/plan.txt") || end failed 70 "the dry run's plan could not be read ($run/plan.txt)"
  hdr="commit=${commit:0:12} date=$start tag=$tag items=$n_items torun=$n_run plan=$plan_s ledger=$ledger run=$run"
  say "$n_items item(s) at ${commit:0:12} (release-list $tag + rotation due), $n_run to run, predicted ${plan_s} s; $(grep -m1 '^gate-batch: budget:' "$run/plan.txt" || echo 'no item past its budget')"
  if [ "$n_run" = 0 ]; then
    # shellcheck disable=SC2086 # the header's fields are words
    python3 -c "$PYNIGHT" report - "$run/night.list" "$NIGHT_HOME/report.tsv" "$run/report.tsv" $hdr state=skip rc=0 > /dev/null ||
      end failed 70 "the skip report could not be written"
    cp "$run/report.tsv" "$NIGHT_HOME/report.tsv.tmp" && mv "$NIGHT_HOME/report.tsv.tmp" "$NIGHT_HOME/report.tsv"
    status skip 0 "every one of the $n_items item(s) is green at its key" "$NIGHT_HOME/report.tsv"
    echo "nightly-real: skip: every one of the $n_items item(s) is green at its key at ${commit:0:12} ($NIGHT_HOME/report.tsv)"
    exit 0
  fi
  if [ "$dry" = --dry-run ]; then
    echo "nightly-real: --dry-run: $n_items item(s), $n_run to run, predicted ${plan_s} s; the list $run/night.list, the plan $run/plan.txt"
    exit 0
  fi
  # 6. the batch
  rc=0
  BLOOMERY_GATE_LEDGER=$ledger BG_IN=/dev/null BG_OUT=$run/batch.out bounded_bg "$NIGHT_BOUND" \
    "$ROOT/tools/gate-batch.sh" --tier real --ledger --over-budget-ok --out "$run/batch" --list "$run/night.list" || rc=$?
  [ "$rc" = 0 ] || end failed 70 "bounded_bg refused the batch's start (rc $rc)"
  bounded_bg_wait || true
  rc=$BG_RC
  [ -n "$BG_SIGNAL" ] && end stopped 143 "stopped by $BG_SIGNAL; the batch's own lines are in $run/batch"
  if ! grep -q '^DONE ' "$run/batch/run.log" 2> /dev/null; then
    case $rc in
      75) end refused 75 "the batch refused to start (rc 75: the lease or a hold came up): $(tail -1 "$run/batch.out"); the previous report stands" ;;
      124 | 137) end failed 70 "the batch ran past BLOOMERY_NIGHT_BOUND ($NIGHT_BOUND s) and was stopped before its DONE line: $run/batch" ;;
      *) end failed 70 "the batch ended rc $rc with no DONE line: $(tail -1 "$run/batch.out")" ;;
    esac
  fi
  # 7. the report
  state=red
  # shellcheck disable=SC2086 # the header's fields are words
  read -r red carried unrun done_line < <(python3 -c "$PYNIGHT" report "$run/batch/run.log" "$run/night.list" "$NIGHT_HOME/report.tsv" \
    "$run/report.tsv" $hdr state=ran rc=$rc) || end failed 70 "the report could not be written from $run/batch/run.log"
  cp "$run/report.tsv" "$NIGHT_HOME/report.tsv.tmp" && mv "$NIGHT_HOME/report.tsv.tmp" "$NIGHT_HOME/report.tsv"
  [ "$red" != 0 ] || state=green
  status "$state" "$([ "$state" = green ] && echo 0 || echo 1)" "$red red ($carried carried), $unrun unrun; $done_line" "$NIGHT_HOME/report.tsv"
  echo "nightly-real: $state: $red red ($carried carried from an earlier night), $unrun unrun of $n_items item(s) at ${commit:0:12}; $done_line ($NIGHT_HOME/report.tsv)"
  [ "$state" = green ] || exit 1
  exit 0
}

# ---- --self-test: this script in a temporary repository beside stub tools (box.sh, gate-batch.sh, recipes.py, rotation.py) and
# the real tools/nightly/units.sh and tools/gate-bound.sh it sources; each case runs the night and reads its rc, lines and report.
self_test() {
  local fails=0 cases=0 rc out
  ST_TMP=$(mktemp -d)
  local t=$ST_TMP
  trap 'rm -rf "$ST_TMP"' EXIT
  st_tree() { # st_tree DIR SCRIPT: a repository at DIR holding SCRIPT as tools/nightly-real.sh and the stubs, tagged v0.0.1
    local d=$1
    mkdir -p "$d/tools/nightly" "$d/tools/ref"
    cp "$2" "$d/tools/nightly-real.sh"
    cp "$ROOT/tools/nightly/units.sh" "$d/tools/nightly/units.sh"
    cp "$ROOT/tools/gate-bound.sh" "$d/tools/gate-bound.sh"
    cp "$ROOT/tools/ref/stub-timeout.py" "$d/tools/ref/stub-timeout.py"
    cat > "$d/tools/box.sh" << 'SH'
#!/usr/bin/env bash
echo "box.sh $*" >> "$STUB_LOG"
if [ "${BLOOMERY_BOX_READONLY:-}" = 1 ]; then exec bash -c "$1"; fi
if [ "${STUB_BOX_RC:-0}" = 75 ]; then echo "[guard] a hold is up: /root/bloomery-peer-hold (owner peer)" >&2; exit 75; fi
exit 0
SH
    cat > "$d/tools/gate-batch.sh" << 'SH'
#!/usr/bin/env bash
echo "gate-batch.sh $* ledger=${BLOOMERY_GATE_LEDGER:-}" >> "$STUB_LOG"
case " $* " in
  *" --classes "*) printf 'gate-a\tF\tbalanced\t3090\tx\n'; exit 0 ;;
  *" --dry-run "*)
    for a in "$@"; do last=$a; done
    while IFS= read -r n; do
      printf 'lane A  %s\n  ledger: %s — stub\n' "$n" "$( [ "${STUB_GREEN:-}" = all ] && echo skip || echo run)"
    done < "$last"
    echo "gate-batch: budget: 1 item(s) past 0.75x the bound that kills them (a real run refuses them, 65, unless --over-budget-ok): gate-c"
    echo "gate-batch: predicted laneA=30s laneB=20s laneX=0s chain=0s wall=31s (derived: stub)"
    exit 0 ;;
esac
out=''
while [ $# -gt 0 ]; do case $1 in --out) out=$2; shift 2 ;; *) shift ;; esac; done
[ "${STUB_BATCH_RC:-}" != 75 ] || { echo "gate-batch: refused: the timing lease is held" >&2; exit 75; }
mkdir -p "$out"
cp "$STUB_RUNLOG" "$out/run.log"
exit "${STUB_BATCH_RC:-1}"
SH
    printf 'import sys\nprint("gate-a\\ngate-b\\ngate-c")\n' > "$d/tools/recipes.py"
    printf 'import sys\na = sys.argv\nopen(a[a.index("--out") + 1], "w").write("# due\\ngate-b\\ngate-d\\n")\n' > "$d/tools/rotation.py"
    chmod +x "$d/tools/box.sh" "$d/tools/gate-batch.sh"
    git -C "$d" init -q
    git -C "$d" add -A
    git -C "$d" -c user.name=selftest -c user.email=selftest@invalid commit -qm selftest
    git -C "$d" tag v0.0.1
  }
  # case NAME WANT_RC WANT_TEXT SCRIPT ENV… : the night in a fresh tree of SCRIPT, run with ENV; its rc and its output must hold
  # WANT_TEXT. Sets out (the night's output) and H (its home).
  st_case() {
    local name=$1 want=$2 text=$3 script=$4
    shift 4
    cases=$((cases + 1))
    rm -rf "$t/c"
    mkdir -p "$t/c"
    st_tree "$t/c/repo" "$script"
    H=$t/c/home
    mkdir -p "$H"
    [ -z "${PREV:-}" ] || printf '%b' "$PREV" > "$H/report.tsv"
    rc=0
    out=$(cd "$t/c/repo" && env STUB_LOG="$t/c/stub.log" BLOOMERY_NIGHT_HOME="$H" BLOOMERY_REMOTE=/nowhere BLOOMERY_NIGHT_BOUND=60 "$@" \
      BLOOMERY_NIGHT_LOCK='' bash tools/nightly-real.sh --in-tree --commit "$(git -C "$t/c/repo" rev-parse HEAD)" ${DRY:+"$DRY"} 2>&1) || rc=$?
    if [ "$rc" != "$want" ] || [[ "$out" != *"$text"* ]]; then
      [ "${QUIET:-0}" = 1 ] || echo "nightly-real self-test: $name: rc $rc (want $want), want '$text'; got: $out" >&2
      fails=$((fails + 1))
      return 1
    fi
  }
  st_want() { # st_want NAME CONDITION…: one more check on the last case
    local name=$1
    shift
    cases=$((cases + 1))
    if ! "$@"; then
      [ "${QUIET:-0}" = 1 ] || echo "nightly-real self-test: $name" >&2
      fails=$((fails + 1))
      return 1
    fi
  }
  # shellcheck disable=SC2329 # st_want runs them by name
  has() { grep -qE "$2" "$1" 2> /dev/null; }
  # shellcheck disable=SC2329
  hasnt() { ! grep -qE "$2" "$1" 2> /dev/null; }
  all_cases() { # all_cases SCRIPT: every case on SCRIPT
    local s=$1 runlog=$t/runlog
    printf '%s\n' "plan laneA=30s laneB=20s wall=31s defaults=0 (predicted, derived from stub)" \
      "gate-a rc=1 12s try=1 lane=A tier=real ledger=red last_green=abc111111111@2026-11-10T00:01:00+0900 tier=real card=3090 range=abc111111111..def222222222 commits=3 touching=[def2222 x]" \
      "gate-b rc=0 5s try=1 lane=B tier=real ledger=pending" "ledger gate-b recorded" \
      "gate-c rc=75 900s try=10 lane=A tier=real" "gate-d rc=skip green-at=abc1234 2026-11-11T00:00:00+0900 lane=A tier=real" \
      "DONE total=4 red=1 tier=real skipped=1 wall=917s plan=31s laneA=912s laneB=5s" > "$runlog"
    PREV='' st_case "skip: every key green" 0 "skip: every one of the 4 item(s) is green at its key" "$s" STUB_GREEN=all &&
      st_want "skip: no batch ran" hasnt "$t/c/stub.log" '^gate-batch.sh --tier real --ledger' &&
      st_want "skip: a skip report" has "$H/report.tsv" '^done	skip: every one of the 4 item' &&
      st_want "skip: the list is release-list + due, sorted, no duplicate" test "$(tr '\n' ' ' < "$(ls -d "$H"/runs/*/night.list)")" = "gate-a gate-b gate-c gate-d "
    PREV='night\tdate=2026-11-11T00:00:00+0900\tcommit=old\nred\tgate-z\trc=1 3s last_green=none range=none\nred\tgate-a\trc=1 old\n' \
      st_case "red: the night names it" 1 "red: 2 red (1 carried from an earlier night), 1 unrun of 4 item(s)" "$s" STUB_RUNLOG="$runlog" \
      BLOOMERY_NIGHT_NOTIFY="bash $t/hook.sh $t/c/told" &&
      st_want "red: the red line holds its last_green and range" has "$H/report.tsv" '^red	gate-a	rc=1 12s .*last_green=abc111111111@2026-11-10T00:01:00\+0900 .*range=abc111111111\.\.def222222222 commits=3' &&
      st_want "red: a red of an earlier night this one gave no verdict is carried" has "$H/report.tsv" '^red	gate-z	rc=1 3s last_green=none range=none carried-from=2026-11-11T00:00:00\+0900$' &&
      st_want "red: a final rc 75 is unrun, not red" has "$H/report.tsv" '^unrun	gate-c	' &&
      st_want "red: a ledger skip and a green are not red" hasnt "$H/report.tsv" '^red	gate-(b|d)	' &&
      st_want "red: the totals line" has "$H/report.tsv" '^done	DONE total=4 red=1 tier=real skipped=1 wall=917s' &&
      st_want "red: the batch runs the real tier on the night's ledger, every item" has "$t/c/stub.log" "^gate-batch.sh --tier real --ledger --over-budget-ok --out .* --list .*night.list ledger=$H/gate-ledger.tsv\$" &&
      st_want "red: the hook is told the state and the report" test "$(cat "$t/c/told" 2> /dev/null)" = "red $H/report.tsv"
    PREV='night\tdate=2026-11-11T00:00:00+0900\tcommit=old\nred\tgate-a\trc=1 old\n' \
      st_case "refusal under a hold" 75 "refused: the box's guard refused (the timing lease or a hold is up): [guard] a hold is up" "$s" STUB_BOX_RC=75 &&
      st_want "refusal: nothing listed, nothing ran" hasnt "$t/c/stub.log" '^gate-batch.sh' &&
      st_want "refusal: the previous report stands" has "$H/report.tsv" '^red	gate-a	rc=1 old$' &&
      st_want "refusal: the status names it" has "$H/status.txt" ' refused rc=75 '
    PREV='' st_case "refusal by the batch" 75 "the batch refused to start (rc 75" "$s" STUB_RUNLOG="$runlog" STUB_BATCH_RC=75 &&
      st_want "batch refusal: no report written" test ! -e "$H/report.tsv"
    PREV='' DRY=--dry-run st_case "dry run" 0 "--dry-run: 4 item(s), 4 to run, predicted 31 s" "$s" STUB_RUNLOG="$runlog" &&
      st_want "dry run: no batch ran" hasnt "$t/c/stub.log" '^gate-batch.sh --tier real --ledger'
  }
  # shellcheck disable=SC2016 # the hook's own text
  printf '%s\n' 'echo "$2 $3" > "$1"' > "$t/hook.sh"
  all_cases "$ROOT/tools/nightly-real.sh"
  # the lock, the outer phase and --pull, on the real script
  rm -rf "$t/o"
  mkdir -p "$t/o/home/lock"
  st_tree "$t/o/repo" "$ROOT/tools/nightly-real.sh"
  sleep 600 &
  local live=$!
  echo "$live" > "$t/o/home/lock/pid"
  cases=$((cases + 1))
  rc=0
  out=$(cd "$t/o/repo" && STUB_LOG=$t/o/stub.log BLOOMERY_NIGHT_HOME=$t/o/home bash tools/nightly-real.sh --commit HEAD 2>&1) || rc=$?
  kill "$live" 2> /dev/null || true
  wait "$live" 2> /dev/null || true
  [ "$rc" = 75 ] && [[ "$out" == *"another night (pid $live) holds"* ]] || { echo "nightly-real self-test: a live lock: rc $rc, want 75 naming it; got: $out" >&2; fails=$((fails + 1)); }
  git init -q --bare "$t/o/origin.git"
  git -C "$t/o/repo" remote add origin "$t/o/origin.git"
  git -C "$t/o/repo" push -q origin HEAD:refs/heads/main
  cases=$((cases + 1))
  rc=0
  out=$(cd "$t/o/repo" && STUB_LOG=$t/o/stub.log STUB_GREEN=all BLOOMERY_NIGHT_HOME=$t/o/home BLOOMERY_REMOTE=/nowhere bash tools/nightly-real.sh 2>&1) || rc=$?
  if [ "$rc" != 0 ] || [[ "$out" != *"the lock $t/o/home/lock names pid $live, which is gone: taken over"* ]] || [[ "$out" != *"skip: every one of the 4 item(s)"* ]] ||
    [ "$(git -C "$t/o/home/tree" rev-parse HEAD 2> /dev/null)" != "$(git -C "$t/o/repo" rev-parse origin/main)" ] || [ -e "$t/o/home/lock" ]; then
    echo "nightly-real self-test: the outer phase (fetch, origin/main, the night's worktree, a stale lock taken over and dropped): rc $rc; got: $out" >&2
    fails=$((fails + 1))
  fi
  cases=$((cases + 1))
  rc=0
  out=$(cd "$t/o/repo" && STUB_LOG=$t/o/stub.log BLOOMERY_NIGHT_PULL=$t/o/home bash tools/nightly-real.sh --pull --into "$t/o/pulled" 2>&1) || rc=$?
  if [ "$rc" != 0 ] || ! cmp -s "$t/o/home/report.tsv" "$t/o/pulled/report.tsv" || [ ! -e "$t/o/pulled/gate-ledger.tsv" ] ||
    ! grep -q '^box.sh cat ' "$t/o/stub.log"; then
    echo "nightly-real self-test: --pull: rc $rc, the report and an empty ledger through read-only box.sh; got: $out" >&2
    fails=$((fails + 1))
  fi
  cases=$((cases + 1))
  rc=0
  out=$(bash "$ROOT/tools/nightly-real.sh" --bogus 2>&1) || rc=$?
  [ "$rc" = 64 ] && [[ "$out" == *"unknown argument --bogus"* ]] || { echo "nightly-real self-test: an unknown argument: rc $rc, want 64; got: $out" >&2; fails=$((fails + 1)); }
  # FAIL-first: each mutant of this script must turn a case red
  mutant() { # mutant NAME OLD NEW
    cases=$((cases + 1))
    python3 -I - "$ROOT/tools/nightly-real.sh" "$t/mutant.sh" "$2" "$3" << 'PY' || { echo "nightly-real self-test: mutant $1: the text is not in the script" >&2; fails=$((fails + 1)); return 0; }
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
i = s.find(old)  # the first occurrence is the code's: the self-test that names it comes after
if i < 0:
    sys.exit(1)
open(dst, "w").write(s[:i] + new + s[i + len(old):])
PY
    local before=$fails
    QUIET=1 all_cases "$t/mutant.sh"
    if [ "$fails" = "$before" ]; then
      echo "nightly-real self-test: mutant $1 left every case green: the cases do not hold it" >&2
      fails=$((fails + 1))
    else
      fails=$before
    fi
  }
  # shellcheck disable=SC2016 # each mutant names the script's own text
  mutant "no skip" '  if [ "$n_run" = 0 ]; then' '  if false; then'
  mutant "no refusal" '    75) end refused 75 "the box'"'"'s guard refused' '    75) : ;; 76) end refused 75 "the box'"'"'s guard refused'
  mutant "the range is dropped" '(m.group(2), ln[len(m.group(1)) + 1:])' '(m.group(2), ln[len(m.group(1)) + 1:].split(" last_green=")[0])'
  mutant "no carried red" '                if not seen and f[1] not in reds:' '                if False:'
  mutant "rc 75 is red" 'NOT_RUN = ("75", "yield")' 'NOT_RUN = ("yield",)'
  if [ "$fails" = 0 ]; then
    echo "nightly-real: self-test ok ($cases cases)"
    return 0
  fi
  echo "nightly-real: self-test: $fails failure(s) in $cases cases" >&2
  return 1
}

# ---- the arguments
MODE=night REV='' DRY='' INTO=${BLOOMERY_NIGHT_DIR:-$HOME/.cache/bloomery/night-pull} IN_TREE=0
while [ $# -gt 0 ]; do
  case $1 in
    --commit) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 64; }; REV=$2; shift 2 ;;
    --dry-run) DRY=--dry-run; shift ;;
    --pull) MODE=pull; shift ;;
    --into) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 64; }; INTO=$2; shift 2 ;;
    --in-tree) IN_TREE=1; shift ;;
    --self-test) [ $# = 1 ] || { echo "$USAGE" >&2; exit 64; }; self_test; exit $? ;;
    *) echo "nightly-real: unknown argument $1; $USAGE" >&2; exit 64 ;;
  esac
done
case $MODE in
  pull) pull "$INTO" ;;
  night)
    mkdir -p "$NIGHT_HOME"
    if [ "$IN_TREE" = 1 ]; then inner "$REV" "$DRY"; else outer "$REV" "$DRY"; fi ;;
esac
