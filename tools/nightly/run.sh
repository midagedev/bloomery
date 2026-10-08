#!/usr/bin/env bash
# The nightly host run: the tests an x86_64 Linux machine with no card and no model files can run, on a machine that is not
# the GPU box. One owner of the set: `python3 tools/recipes.py host-nightly` (crates and check scripts, the exclusions by name
# with reasons in `--excluded`); this script only runs the units it prints and records the outcome.
#
# Installed by tools/nightly/install.sh to /usr/local/lib/bloomery-nightly/run.sh (root-owned, so a pushed change to this
# file cannot silence the alarm) and started by bloomery-nightly.service as the unprivileged `bnightly` user. The mail
# is not sent here: bnightly has no credentials; tools/nightly/mail.sh runs as the unit's ExecStopPost and decides from the
# files this script leaves.
#
# A run:  lock -> fetch origin/main and hard-reset the clone -> toolchain from rust-toolchain.toml -> the units in order,
# each under its own bound, output to <run>/log and <run>/units/<unit>.log -> summary, units.tsv, failed, first-failure.tail,
# status, rc -> keep the last 14 runs, point `latest` at this one.
#
# Files of <run> = $STATE/<YYYY-MM-DD in Asia/Seoul>, or <date>-<n> for a second run the same day:
#   summary             one line: date, commit, tree, GREEN|RED, wall, then each unit's counts; RUNNING until the run ends
#   status              GREEN | RED | RUNNING
#   rc                  0 green; 1 a unit failed; 2 the run could not start its units (fetch, toolchain, owner)
#   commit              the short sha tested, `+local` when the working tree holds uncommitted files
#   units.tsv           unit, kind (cargo|script), rc, passed, failed, ignored, seconds
#   failed              one line per failing test, and one per unit that failed with no failing test (build, timeout, script)
#   first-failure.tail  the last 80 lines of the first failing unit's log
#   excluded.tsv        what the set leaves out, with why (`host-nightly --excluded`)
#   invocation          the systemd invocation id, which mail.sh matches
#
# Exit: 0 green, 1 red, 2 could not run the units, 75 another run holds the lock (nothing was run).
#
# Environment (all optional):
#   NIGHTLY_HOME         /srv/bloomery-nightly: clone in repo/, CARGO_TARGET_DIR target/, rustup and cargo homes, opt/cuda-13.3, bin/just
#   NIGHTLY_STATE        /var/lib/bloomery-nightly
#   NIGHTLY_REPO         https://github.com/midagedev/bloomery.git (public HTTPS, no credentials)
#   NIGHTLY_NO_FETCH=1   test the clone's working tree as it is (no fetch, no reset); the summary says `local`. For proving a change before it is pushed.
#   NIGHTLY_INJECT_FAIL=1  add one deliberately failing unit, `inject-fail`, that prints a failing libtest line: the red path end to end
#   NIGHTLY_UNIT_BOUND   seconds one unit may take (default 1200); it is BLOOMERY_GATE_BOUND for tools/gate.sh
#
# This machine's differences from the box, stated: RUSTFLAGS is `-C target-cpu=x86-64-v3` here (the VPS guest is a Zen 1 EPYC; the
# repo's .cargo/config.toml names znver3, which RUSTFLAGS overrides), bindgen reads libclang 18 (the box has LLVM 21), the
# CUDA headers are the 13.3 headers only (no driver, no toolkit), and no model file or $BLOOMERY_DATA exists.
set -uo pipefail

HOME_DIR=${NIGHTLY_HOME:-/srv/bloomery-nightly}
STATE=${NIGHTLY_STATE:-/var/lib/bloomery-nightly}
REPO_URL=${NIGHTLY_REPO:-https://github.com/midagedev/bloomery.git}
UNIT_BOUND=${NIGHTLY_UNIT_BOUND:-1200}
KEEP=14
CLONE=$HOME_DIR/repo

export HOME=$HOME_DIR
export PATH=$HOME/.cargo/bin:$HOME/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_TOOLKIT_PATH=$HOME/opt/cuda-13.3
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
export CARGO_TARGET_DIR=$HOME/target
export CARGO_TERM_COLOR=never
export RUSTFLAGS='-C target-cpu=x86-64-v3'
export BLOOMERY_GATE_BOUND=$UNIT_BOUND
unset BLOOMERY_BOX_ENV BLOOMERY_DATA

# Atomic one-line file write, so mail.sh never reads half a file.
put() { printf '%s\n' "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }

[ -d "$STATE" ] || { echo "nightly: no state dir $STATE (tools/nightly/install.sh makes it)" >&2; exit 2; }
exec 9> "$STATE/.lock"
flock -n 9 || { echo "nightly: another run holds $STATE/.lock; nothing was run" >&2; exit 75; }

DATE=$(TZ=Asia/Seoul date +%F)
RUNID=$DATE
n=1
while [ -e "$STATE/$RUNID" ]; do n=$((n + 1)); RUNID=$DATE-$n; done
RUN=$STATE/$RUNID
mkdir -p "$RUN/units" || exit 2
put "$RUN/invocation" "${INVOCATION_ID:-manual}"
put "$RUN/status" RUNNING
put "$RUN/summary" "$DATE ? ? RUNNING (started $(TZ=Asia/Seoul date +%H:%M:%S) KST)"
put "$STATE/current" "$RUNID"
exec >> "$RUN/log" 2>&1
START=$SECONDS
COMMIT="?"
TREE_LABEL="?"
CURRENT_UNIT=""
DRIFT=""

finish() {
  # finish <rc> <GREEN|RED> <tail of the summary line>
  local rc=$1 status=$2 rest=$3 wall=$((SECONDS - START))
  put "$RUN/rc" "$rc"
  put "$RUN/status" "$status"
  put "$RUN/summary" "$DATE $COMMIT $TREE_LABEL $status wall=${wall}s $rest${DRIFT:+ | DRIFT: installed$DRIFT differ from the tree}"
  echo "== $status rc=$rc wall=${wall}s"
  # keep the last $KEEP runs, newest by name; `latest` points at this one
  local d i=0
  for d in $(cd "$STATE" && ls -1d [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]* 2> /dev/null | LC_ALL=C sort -r); do
    i=$((i + 1))
    if [ "$i" -gt "$KEEP" ] && [ -d "$STATE/$d" ] && [ ! -L "$STATE/$d" ]; then rm -rf "${STATE:?}/$d"; fi
  done
  ln -sfn "$RUNID" "$STATE/latest.tmp" && mv -fT "$STATE/latest.tmp" "$STATE/latest"
  find "$STATE" -maxdepth 1 -name 'heartbeat-*' -mtime +30 -delete
  exit "$rc"
}

interrupted() {
  echo "== interrupted by $1 during ${CURRENT_UNIT:-start}"
  put "$RUN/failed" "run: interrupted by $1 during ${CURRENT_UNIT:-start} (a bound or a stop of the service)"
  finish 1 RED "interrupted by $1 during ${CURRENT_UNIT:-start}"
}
trap 'interrupted TERM' TERM
trap 'interrupted INT' INT

echo "== nightly run $RUNID, $(date -u +%FT%TZ), host $(uname -nr), $(nproc) cpus, CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS, unit bound ${UNIT_BOUND}s"

# ---- the tree ----
if [ "${NIGHTLY_NO_FETCH:-}" = 1 ]; then
  [ -d "$CLONE/.git" ] || { echo "nightly: NIGHTLY_NO_FETCH=1 and no clone at $CLONE"; put "$RUN/failed" "run: no clone at $CLONE"; finish 2 RED "no clone"; }
  TREE_LABEL=local
else
  if [ ! -d "$CLONE/.git" ]; then
    git clone --quiet "$REPO_URL" "$CLONE" || { put "$RUN/failed" "run: git clone failed"; finish 2 RED "git clone failed"; }
  fi
  timeout 300 git -C "$CLONE" fetch --quiet --prune origin main &&
    git -C "$CLONE" reset --quiet --hard origin/main || { put "$RUN/failed" "run: fetch or reset of $CLONE failed"; finish 2 RED "fetch failed"; }
  TREE_LABEL=origin/main
fi
COMMIT=$(git -C "$CLONE" rev-parse --short=8 HEAD)
if [ -n "$(git -C "$CLONE" status --porcelain --untracked-files=normal)" ]; then COMMIT=$COMMIT+local; fi
put "$RUN/commit" "$COMMIT"
put "$RUN/summary" "$DATE $COMMIT $TREE_LABEL RUNNING (started $(TZ=Asia/Seoul date +%H:%M:%S) KST)"
echo "== tree $TREE_LABEL $COMMIT"
cd "$CLONE" || finish 2 RED "no clone"

# The installed scripts are copies: a later edit of the tree's own scripts reaches them only through install.sh. Say so, loudly enough
# to be read in the summary (a green run mails nothing).
for f in run.sh mail.sh; do
  if [ -f "tools/nightly/$f" ] && ! cmp -s "tools/nightly/$f" "/usr/local/lib/bloomery-nightly/$f"; then DRIFT="$DRIFT $f"; fi
done
if [ -n "$DRIFT" ]; then echo "== NOTE: installed${DRIFT} differ from the tree's tools/nightly/ copies: run tools/nightly/install.sh"; fi

# ---- the toolchain: the channel rust-toolchain.toml pins, minimal profile (no components a test needs) ----
CHANNEL=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml | head -1)
if [ -z "$CHANNEL" ]; then put "$RUN/failed" "run: no channel in rust-toolchain.toml"; finish 2 RED "no toolchain channel"; fi
export RUSTUP_TOOLCHAIN=$CHANNEL
if ! rustup toolchain list | grep -q "^$CHANNEL"; then
  echo "== installing toolchain $CHANNEL"
  rustup toolchain install "$CHANNEL" --profile minimal --no-self-update || { put "$RUN/failed" "run: rustup could not install $CHANNEL"; finish 2 RED "toolchain install failed"; }
  for t in $(rustup toolchain list | awk '{print $1}' | grep -v "^$CHANNEL"); do rustup toolchain uninstall "$t"; done
fi
echo "== $(rustc --version), $(cargo --version), $(just --version 2> /dev/null || echo 'no just')"

# ---- the units: the owner's list ----
PLAN=$RUN/plan.tsv
python3 tools/recipes.py host-nightly > "$PLAN" || { put "$RUN/failed" "run: tools/recipes.py host-nightly failed (the owner of the set)"; finish 2 RED "owner failed"; }
python3 tools/recipes.py host-nightly --excluded > "$RUN/excluded.tsv" || { put "$RUN/failed" "run: host-nightly --excluded failed"; finish 2 RED "owner failed"; }
if [ "${NIGHTLY_INJECT_FAIL:-}" = 1 ]; then
  # the red path end to end: a failing libtest line the parser must name, and a nonzero rc
  printf 'inject-fail\t%s\n' "echo 'test injected::deliberate_failure ... FAILED'; echo 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out'; exit 101" >> "$PLAN"
  echo "== NIGHTLY_INJECT_FAIL=1: unit inject-fail added"
fi
if [ ! -s "$PLAN" ]; then put "$RUN/failed" "run: the owner printed no unit"; finish 2 RED "no units"; fi
echo "== left out by name:"
sed 's/^/   /' "$RUN/excluded.tsv"
: > "$RUN/units.tsv"
: > "$RUN/failed"

# ---- run ----
COUNTS=""
FAILS=0
FIRST_FAIL=""
while IFS=$'\t' read -r name cmd; do
  CURRENT_UNIT=$name
  ulog=$RUN/units/$name.log
  echo "== unit $name: $cmd"
  t0=$SECONDS
  timeout --kill-after=10 "$((UNIT_BOUND + 30))" bash -c "$cmd" < /dev/null 2>&1 | tee "$ulog"
  rc=${PIPESTATUS[0]}
  secs=$((SECONDS - t0))
  kind=script
  case $cmd in *gate.sh*) kind=cargo ;; esac
  read -r p f i < <(awk '/^test result: / { for (k = 2; k <= NF; k++) { if ($k == "passed;") p += $(k - 1); if ($k == "failed;") f += $(k - 1); if ($k == "ignored;") g += $(k - 1) } } END { print p + 0, f + 0, g + 0 }' "$ulog")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$rc" "$p" "$f" "$i" "$secs" >> "$RUN/units.tsv"
  if [ "$kind" = cargo ] || [ "$name" = inject-fail ]; then COUNTS="$COUNTS $name $p/$f/$i"; else COUNTS="$COUNTS $name $([ "$rc" = 0 ] && echo ok || echo "rc$rc")"; fi
  if [ "$rc" != 0 ]; then
    FAILS=$((FAILS + 1))
    [ -n "$FIRST_FAIL" ] || FIRST_FAIL=$ulog
    named=$(sed -n 's/^test \(.*\) \.\.\. FAILED$/\1/p' "$ulog")
    if [ -n "$named" ]; then
      printf '%s\n' "$named" | sed "s/^/$name: /" >> "$RUN/failed"
    else
      case $rc in
        124 | 137) why="timeout after ${UNIT_BOUND}s (rc $rc)" ;;
        78) why="a test filter matched no test (rc 78)" ;;
        *) if grep -q 'could not compile\|^error\(\[E[0-9]*\]\)\?:' "$ulog"; then why="build failure (rc $rc)"; else why="rc $rc"; fi ;;
      esac
      echo "$name: $why" >> "$RUN/failed"
      # a script's red lines sit anywhere in its output, not in its last 80: name the first few
      grep -E 'FAIL|[Ff]ailed|[Ee]rror' "$ulog" | grep -v '^ok ' | head -n 8 | cut -c1-220 | sed 's/^/    | /' >> "$RUN/failed"
    fi
  fi
done < "$PLAN"
CURRENT_UNIT=""

[ -z "$FIRST_FAIL" ] || tail -n 80 "$FIRST_FAIL" > "$RUN/first-failure.tail"
UNITS=$(wc -l < "$PLAN" | tr -d ' ')
if [ "$FAILS" = 0 ]; then
  finish 0 GREEN "units=$UNITS |$COUNTS"
else
  finish 1 RED "units=$UNITS failed=$FAILS |$COUNTS"
fi
