#!/usr/bin/env bash
# The model load's wall under the lease: one start per arm per round, each stopped at its first `load phases`
# record (run on the box).
#
#   bash tools/ref/load-ab.sh [--warm] [--bin NAME] <remote-dir>... [-- <the bin's arguments>]
#
# What it blocks: a load-path change judged by one start's wall beside another window's, or on a page cache the
# previous start filled. Every <remote-dir> is a name under ~/repo/ whose target/release/<bin> was built there
# (cargo oxide build ... --bin <bin>, in that tree); the first is the base, and every later arm is compared with it.
# <bin> is generate_ds41 unless --bin names another binary of a schema that prints `load phases`; with another
# binary the arguments after `--` are required, with generate_ds41 they default to `--place a -n 1 --tokens 100000`
# (plan (a), the one id the load needs to be asked for something). The arms inherit the runner's environment, so a
# lever a comparison holds fixed goes through BLOOMERY_BOX_ENV.
#
# A start. Cold (the default): every shard of the model file is dropped from the page cache (posix_fadvise
# DONTNEED, then mincore: tools/ref/evict.sh) and a start whose shards stay more than 1 % resident is a FAIL row
# `rc=evict`, not run. --warm skips the drop: the restart, the page cache as the previous start left it. The
# binary runs under the arm bound (BLOOMERY_ARM_BOUND) and is stopped one second after its `load phases` record
# appears (the records the load prints right after it, `residency host` among them, are in the log by then); the
# stop is a TERM by the pid the runner holds (tools/gate-bound.sh bounded_bg_stop), and its wall is on the row.
# One discarded start per arm opens the sitting (`WARMUP r0` rows, never evicted, in no mean): the driver's JIT
# cache for a binary just built, and under --warm the page cache. Rounds run every arm once, the order rotated a
# slot each round (ab-decode.sh's reason: the first arm of a round reads slow).
#
# A row: every field of the `load phases` record (seconds), `load_s` of the `load` record, the wall from the
# spawn to the record (polled every 0.2 s), the stop's wall, the eviction's wall and resident fraction, and the
# witness over the start: the change in pgmajfault, in pgpgin (kB the kernel paged in, GiB on the row) and in `Cached`
# (/proc/meminfo), the last two read just before the spawn and just after the stop. The row ends in [cpu-busy] /
# [other-busy] when guard_cpu / guard_other tagged it. After the rounds, per field, each later arm against the base
# round by round: the mean of new/base with its 95 % interval (Student t at rounds - 1 degrees of freedom, as
# depth-ds41.sh's ratio table), the same as an effect in percent, and the arms' means.
# An arm that fails (eviction, the binary dying or running past the bound, an unreadable record) is a FAIL row and
# the runner goes on; a round whose pair lost an arm drops out of that pair's means, and the run ends rc 1.
#
# The card is required (BLOOMERY_LEASE_CARD), the rounds are BLOOMERY_AB_ROUNDS (default 3), and the lease's
# timing card is the GPU runners' (tools/ref/timing-card.sh). `--self-test` runs on any machine with stub arms.
set -uo pipefail
HERE=${BASH_SOURCE[0]%/*}
[ "$HERE" != "${BASH_SOURCE[0]}" ] || HERE=.
# MODEL: the profile's file (BLOOMERY_REF_MODEL overrides it; tools/box.sh exports it, so the profile is picked
# there: BLOOMERY_MODEL=deepseek41 tools/box.sh ...).
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
# The lease, the witness fields and bounded_bg (tools/gate-bound.sh, through lease.sh).
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"
# shellcheck source=tools/ref/evict.sh
source "$HERE/evict.sh"

# The seconds the kernel may take to unmap a stopped binary after the stop's own 9 s budget (tools/gate-bound.sh) ran
# out and sent the KILL: the host set a V4.1 start holds is locked file pages. A binary still alive after them ends
# the runner (70), which holds the lease for as long as that process lives.
UNMAP_WAIT=${UNMAP_WAIT:-120}
# The seconds between the `load phases` record and the stop: the records the load prints right after it are in the log.
RECORD_GRACE=${RECORD_GRACE:-1}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
# Box minutes per round for the card's ruler refusal to price: each arm's eviction, start and stop, about a minute.
ROUND_MINUTES=${ROUND_MINUTES:-4}
# The first `load phases` field names, in the record's order; plan_s is optional (a loader that times no plan).
PHASES=(open plan context upload derive host_set body head other)
# A page-cache residency above this percent of the file's pages after the drop fails the start.
EVICT_RESIDENT_PCT=1
# The row tags guard_cpu and guard_other set (timing-card.sh defines the second where it is sourced).
CPU_BUSY_TAG='' OTHER_BUSY_TAG=${OTHER_BUSY_TAG:-}
SHAS=()
WITNESS=(head-open indent card busiest model meminfo pgmajfault)

usage() {
  echo "usage: load-ab.sh [--warm] [--bin NAME] <remote-dir> <remote-dir>... [-- <the bin's arguments>]" >&2
  exit 64
}

# clock: seconds with microseconds (bash 5's EPOCHREALTIME), whole seconds where the shell has none (the Mac's self-test).
clock() {
  if [ -n "${EPOCHREALTIME:-}" ]; then echo "${EPOCHREALTIME/,/.}"; else date +%s; fi
}
# secs <t0> <t1>: t1 - t0 in seconds, two decimals.
secs() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", b - a }'; }
# vm_sample: `<pgmajfault> <pgpgin kB> <Cached kB>` now.
vm_sample() {
  awk '$1 == "pgmajfault" { m = $2 } $1 == "pgpgin" { p = $2 } END { printf "%s %s ", m, p }' /proc/vmstat
  awk '/^Cached:/ { print $2 }' /proc/meminfo
}
# alive <pid>: running, not a zombie (kill -0 answers for a zombie the shell has not waited for).
alive() { local st; st=$(ps -o stat= -p "$1" 2> /dev/null) || return 1; case $st in '' | *Z*) return 1 ;; esac; }

# fail_row <round> <label> <rc> <why>: a FAIL row in place of the start's row; the start is out of the means.
FAILED=()
fail_row() {
  echo "FAIL r$1 $2 rc=$3 | $4$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  FAILED+=("r$1 $2 rc=$3")
}

# evict_start <round> <label>: the cold start's drop, its witness line in EV_LINE; 1 (a FAIL row printed) when the
# advice did not run or more than EVICT_RESIDENT_PCT percent of the pages stay resident.
evict_start() {
  local t0 t1 pct
  t0=$(clock)
  if ! evict_set "$MODEL"; then
    fail_row "$1" "$2" evict "the advice over $MODEL's $EVICT_SHARDS shard(s) did not run: $EVICT_ERR"
    return 1
  fi
  t1=$(clock)
  pct=$(awk -v a="$EVICT_AFTER" -v p="$EVICT_PAGES" 'BEGIN { printf "%.2f", (p > 0) ? 100 * a / p : 0 }')
  EV_S=$(secs "$t0" "$t1") EV_PCT=$pct
  echo "    evict: $MODEL shards=$EVICT_SHARDS pages=$EVICT_PAGES resident before=$EVICT_BEFORE after=$EVICT_AFTER ($pct %) in $EV_S s (posix_fadvise DONTNEED, then mincore)"
  if awk -v a="$pct" -v m="$EVICT_RESIDENT_PCT" 'BEGIN { exit !(a > m) }'; then
    fail_row "$1" "$2" evict "evict: $pct % of the $EVICT_PAGES pages of $MODEL's $EVICT_SHARDS shard(s) stay resident after posix_fadvise(DONTNEED), over the $EVICT_RESIDENT_PCT % the runner allows (a page another process holds mapped is not evicted): the start would read them warm"
    return 1
  fi
}

# sha_of <label>: the arm's binary, as check_bins read it (`-` when it did not).
sha_of() {
  local i
  for i in "${!DIRS[@]}"; do
    if [ "${DIRS[$i]}" = "$1" ]; then echo "${SHAS[$i]:--}"; return 0; fi
  done
  echo -
}

# start_arm <ROW|WARMUP> <round> <label>: one start. A ROW's fields go to RECS (`field|label|round|value`) and the row
# prints; a start that failed prints its FAIL row and returns 1.
RECS=()
start_arm() {
  local tag=$1 r=$2 label=$3 bin=$HOME/repo/$3/target/release/$BIN_NAME
  local out=$WORK/$tag-r$r-$label.log pid seen=0 rc v0 v1 t0 t1 t2 f specs row='' ph k
  CPU_BUSY_TAG='' EV_S=- EV_PCT=-
  if [ "$WARM" = 0 ] && [ "$tag" = ROW ]; then evict_start "$r" "$label" || return 1; fi
  guard_other
  guard_cpu "pre $tag r$r $label"
  witness "pre $tag r$r $label"
  read -r -a v0 <<< "$(vm_sample)"
  t0=$(clock)
  BG_IN=/dev/null BG_OUT=$out bounded_bg "$LEASE_ARM_BOUND" "$bin" "${ARGS[@]}" || { fail_row "$r" "$label" 70 "bounded_bg refused the start"; return 1; }
  pid=$(bounded_bg_get PID)
  while alive "$pid"; do
    if grep -q '^load phases ' "$out" 2> /dev/null; then seen=1; break; fi
    sleep 0.2 9>&-
  done
  t1=$(clock)
  if [ "$seen" = 0 ] && grep -q '^load phases ' "$out" 2> /dev/null; then seen=1; fi
  if alive "$pid"; then
    [ "$seen" = 0 ] || sleep "$RECORD_GRACE" 9>&-
    if ! bounded_bg_stop; then
      k=0
      while alive "$pid" && [ "$k" -lt "$UNMAP_WAIT" ]; do sleep 1 9>&-; k=$((k + 1)); done
      if alive "$pid"; then echo "load-ab.sh: $label's start (pid $pid) is still alive $UNMAP_WAIT s after the KILL: the lease is held by it, the runner ends" >&2; exit 70; fi
    fi
  else
    bounded_bg_wait
  fi
  t2=$(clock)
  rc=$(bounded_bg_get RC)
  read -r -a v1 <<< "$(vm_sample)"
  witness "post $tag r$r $label"
  guard_cpu "post $tag r$r $label"
  if [ "$seen" = 0 ]; then
    fail_row "$r" "$label" "$rc" "$BIN_NAME ended or ran out of its bound with no \`load phases\` record; last line: $(grep -v '^[[:space:]]*$' "$out" | tail -n 1)"
    tail -n 20 "$out" >&2
    return 1
  fi
  specs=("LOAD_S=load.load_s")
  for ph in "${PHASES[@]}"; do specs+=("P_$ph=load_phases.${ph}_s"); done
  if ! eval "$(python3 -I "$HERE/../bloomery/records.py" sh --bin "$BIN_NAME" "$out" "${specs[@]}" 2> "$WORK/records.err")" || [ -z "${LOAD_S:-}" ]; then
    fail_row "$r" "$label" 65 "the \`load\` and \`load phases\` records of $BIN_NAME's output did not read: $(tail -n 1 "$WORK/records.err")"
    return 1
  fi
  for ph in "${PHASES[@]}"; do
    f=P_$ph
    if [ -z "${!f}" ] && [ "$ph" != plan ]; then
      fail_row "$r" "$label" 65 "the \`load phases\` record has no ${ph}_s"
      return 1
    fi
    row+="${ph}_s ${!f:--} | "
  done
  row="load_s $LOAD_S | ${row}wall_s $(secs "$t0" "$t1") | stop_s $(secs "$t1" "$t2") | evict_s $EV_S resident_pct $EV_PCT | majflt $((v1[0] - v0[0])) | pgpgin_gib $(awk -v a="${v0[1]}" -v b="${v1[1]}" 'BEGIN { printf "%.2f", (b - a) / 1048576 }') | cached_gib $(awk -v a="${v0[2]}" -v b="${v1[2]}" 'BEGIN { printf "%.1f->%.1f", a / 1048576, b / 1048576 }') | binary $(sha_of "$label")"
  echo "$tag r$r $label | $row$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  grep -E '^(place |plan |host_populate=|load |load phases |residency host )' "$out" | sed 's/^/    /' || true
  [ "$tag" = ROW ] || return 0
  RECS+=("load_s|$label|$r|$LOAD_S")
  for ph in "${PHASES[@]}"; do
    f=P_$ph
    [ -z "${!f}" ] || RECS+=("${ph}_s|$label|$r|${!f}")
  done
  RECS+=("wall_s|$label|$r|$(secs "$t0" "$t1")")
}

# effects <base label> <label>...: per field, each later arm against the base, from RECS. Each round's new/base
# ratio (a round either arm lost has none), their mean and its 95 % interval, the arms' means.
effects() {
  [ "${#RECS[@]}" -gt 0 ] || return 0
  printf '%s\n' "${RECS[@]}" | awk -F'|' -v base="$1" -v others="${*:2}" -v t975="$T975" '
    { f = $1; if (!(f in seen)) { seen[f] = 1; order[++nf] = f }
      v[f SUBSEP $2 SUBSEP $3] = $4; if ($3 + 0 > nr) nr = $3 + 0 }
    END {
      nt = split(t975, t, " "); no = split(others, o, " ")
      for (i = 1; i <= nf; i++) for (j = 1; j <= no; j++) {
        f = order[i]; c = 0; m = 0; list = ""; sb = 0; sn = 0
        for (r = 1; r <= nr; r++) {
          kb = f SUBSEP base SUBSEP r; kn = f SUBSEP o[j] SUBSEP r
          if (!(kb in v) || !(kn in v) || v[kb] + 0 == 0) continue
          q = v[kn] / v[kb]; c++; x[c] = q; m += q; sb += v[kb]; sn += v[kn]
          list = list sprintf(" r%d %.4f", r, q)
        }
        if (c == 0) continue
        m /= c; ss = 0
        for (k = 1; k <= c; k++) ss += (x[k] - m) ^ 2
        if (c < 2) ci = "(one round: no interval)"
        else if (c - 1 > nt) ci = sprintf("(no t quantile for df %d)", c - 1)
        else { h = t[c - 1] * sqrt(ss / (c - 1)) / sqrt(c); ci = sprintf("%+.1f %% +- %.1f %%", (m - 1) * 100, h * 100) }
        printf "effect %-10s %s/%s  means %.2f -> %.2f s  ratio %.4f  effect %s (n=%d)  per round:%s\n", f, o[j], base, sb / c, sn / c, m, ci, c, list
      }
    }'
}

# sitting: the lease, the warm-up starts, the rounds, the effects. Reads DIRS, ARGS, WARM, BIN_NAME and ROUNDS.
sitting() {
  local r i n=${#DIRS[@]} a w0 w1 warmups=0 timed
  lease_take
  w0=$(clock)
  lease_out_init || exit 70
  WORK=$LEASE_OUT_DIR
  timed=$((ROUNDS * n))
  echo "[config] bin=$BIN_NAME args=${ARGS[*]} model=$MODEL tier=$([ "$WARM" = 1 ] && echo warm || echo cold) rounds=$ROUNDS arms=${DIRS[*]} base=${DIRS[0]}"
  echo "[config] starts: $n warm-ups (never evicted, in no mean) + $timed timed$([ "$WARM" = 1 ] || echo " ($timed evictions of $MODEL's shards)"); the arm bound $LEASE_ARM_BOUND s, the unmap wait after a failed stop $UNMAP_WAIT s"
  echo "[config] env: $(env | grep -E '^(BLOOMERY_|CUDA_VISIBLE_DEVICES=)' | grep -v -E '^BLOOMERY_(LEASE_CARD|AB_ROUNDS|REF_MODEL)=' | sort | tr '\n' ' ')"
  witness pre
  guard_other
  for a in "${DIRS[@]}"; do
    start_arm WARMUP 0 "$a" || :
    warmups=$((warmups + 1))
  done
  for r in $(seq "$ROUNDS"); do
    for i in $(seq 0 $((n - 1))); do
      start_arm ROW "$r" "${DIRS[$(((i + r - 1) % n))]}" || :
    done
  done
  echo
  echo "=== paired effect of each arm against ${DIRS[0]} (new / base per round; mean ratio with its 95 % interval; a start that failed drops its round from that pair) ==="
  effects "${DIRS[@]}"
  [ "${#FAILED[@]}" = 0 ] || echo "failed starts: $(printf '%s; ' "${FAILED[@]}")"
  witness post
  w1=$(clock)
  echo "[done] sitting wall $(secs "$w0" "$w1") s, $((warmups + timed)) starts, ${#FAILED[@]} failed"
  [ "${#FAILED[@]}" = 0 ]
}

parse_args() {
  WARM=0 BIN_NAME=generate_ds41 DIRS=() ARGS=()
  while [ $# -gt 0 ]; do
    case $1 in
      --warm) WARM=1 ;;
      --bin) [ $# -ge 2 ] || usage; BIN_NAME=$2; shift ;;
      --) shift; ARGS=("$@"); break ;;
      -*) usage ;;
      *) DIRS+=("$1") ;;
    esac
    shift
  done
  [ "${#DIRS[@]}" -ge 2 ] || { echo "load-ab.sh: a base and one arm at least, got ${#DIRS[@]} remote dir(s)" >&2; usage; }
  if [ "${#ARGS[@]}" = 0 ]; then
    if [ "$BIN_NAME" != generate_ds41 ]; then
      echo "load-ab.sh: --bin $BIN_NAME needs its arguments after -- (the defaults are generate_ds41's)" >&2
      exit 64
    fi
    ARGS=(--place a -n 1 --tokens 100000)
  fi
}

# check_bins: every arm's binary is there and newer than its sources, else the named refusal (2, 3).
check_bins() {
  local d b rc
  for d in "${DIRS[@]}"; do
    b=$HOME/repo/$d/target/release/$BIN_NAME
    [ -x "$b" ] || { echo "load-ab.sh: no $b: cargo oxide build the bin in that tree first" >&2; exit 2; }
    rc=0
    (cd "$HOME/repo/$d" && assert_fresh_binary "target/release/$BIN_NAME") || rc=$?
    [ "$rc" = 0 ] || exit "$rc"
    SHAS+=("$(sha256sum "$b" | cut -c1-12)")
  done
}

self_test() {
  local t fails=0 up load k wall n
  t=$(mktemp -d "${TMPDIR:-/tmp}/load-ab-self-test.XXXXXX") || exit 70
  # shellcheck disable=SC2064 # the directory now
  trap "rm -rf '$t'" EXIT
  check() { if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi; }
  matches() { if printf '%s\n' "$3" | grep -E -- "$2" > /dev/null; then echo "ok $1"; else echo "FAIL $1: no line matches /$2/ in [$3]"; fails=$((fails + 1)); fi; }
  count() { n=$(printf '%s\n' "$2" | grep -c -E -- "$3" || true); check "$1" "$n" "$4"; }
  # shellcheck source=tools/ref/cancel-stubs.sh
  source "$HERE/cancel-stubs.sh"
  mkdir "$t/tbin" "$t/home"
  command -v timeout > /dev/null 2>&1 || { cancel_timeout "$t/tbin/timeout"; chmod +x "$t/tbin/timeout"; export CANCEL_TIMEOUT_FORWARD=1; }
  PATH="$t/tbin:$PATH"
  # The stub arms. Each prints the records of a load, valid by the schema (records.py's own sample line with this
  # arm's numbers in it), then waits for the TERM; `die` ends with rc 3 before any record.
  for k in base new; do
    case $k in base) up=10.00 load=20.5 ;; *) up=2.50 load=12.0 ;; esac
    python3 -I -c '
import sys
sys.path.insert(0, sys.argv[1])
import records
schema = records.load("generate_ds41")
load = records.sample_line(schema.by_name["load"]).replace(" in 1.5 s ", " in " + sys.argv[3] + " s ")
phases = records.sample_line(schema.by_name["load_phases"])
for k, v in (("open", "0.10"), ("plan", "0.30"), ("context", "0.20"), ("upload", sys.argv[2]), ("derive", "0.40"),
             ("host_set", "1.50"), ("body", "1.00"), ("head", "0.10"), ("other", "0.20")):
    phases = phases.replace(" " + k + "_s=1.5", " " + k + "_s=" + v)
print("place unset a")
print(load)
print(phases)
' "$HERE/../bloomery" "$up" "$load" > "$t/lines-$k"
    mkdir -p "$t/home/repo/$k/target/release"
    printf '#!/bin/sh\ncat %s\nexec sleep 60\n' "$t/lines-$k" > "$t/home/repo/$k/target/release/generate_ds41"
  done
  mkdir -p "$t/home/repo/die/target/release"
  printf '#!/bin/sh\necho "loading..."\nexit 3\n' > "$t/home/repo/die/target/release/generate_ds41"
  chmod +x "$t"/home/repo/*/target/release/generate_ds41
  # sit <K=V>... -- <load-ab.sh arguments>: the sitting in a subshell on the stubs of what a box gives (no card, no
  # lease, no witness, no cpu or card guard, a drop that leaves STUB_RESIDENT of 1000 pages); its output in OUT, its rc
  # in RC.
  # shellcheck disable=SC2030,SC2031 # the sitting's variables live in its subshell, never the self-test's
  sit() {
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    RC=0
    OUT=$({
      trap - EXIT # the self-test's own cleanup is not this sitting's
      for k in ${envs[@]+"${envs[@]}"}; do export "${k?}"; done
      export HOME=$t/home
      RECORD_GRACE=0.1
      lease_take() { :; }
      witness() { echo "witness $*"; }
      guard_other() { OTHER_BUSY_TAG=''; }
      guard_cpu() { :; }
      vm_sample() { echo "$((100 + ${STUB_FAULTS:-0})) $((1000 + ${STUB_PGPGIN:-0})) 2000000"; }
      evict_set() {
        EVICT_SHARDS=2 EVICT_PAGES=1000 EVICT_BEFORE=900 EVICT_AFTER=${STUB_RESIDENT:-0} EVICT_ERR=boom
        [ -z "${STUB_NOADVICE:-}" ]
      }
      MODEL=/stub/model-00001-of-00002.gguf
      ROUNDS=${STUB_ROUNDS:-3}
      T975=$(python3 "$HERE/tdist.py" "$ROUNDS")
      parse_args "$@"
      sitting
    } 2>&1) || RC=$?
  }
  # a: two arms, three rounds, cold: both warm-ups first and unevicted, six evicted rows in the rotated order, the
  # stub's 60 s wait cut at the record (the case inside a few seconds), the upload effect -75 %.
  wall=$(date +%s)
  sit STUB_ROUNDS=3 -- base new -- --any
  wall=$(($(date +%s) - wall))
  check a-rc "$RC" 0
  if [ "$wall" -lt 40 ]; then echo "ok a-stopped-at-the-record ($wall s)"; else echo "FAIL a-stopped-at-the-record: $wall s"; fails=$((fails + 1)); fi
  matches a-config '^\[config\] bin=generate_ds41 args=--any model=/stub/model-00001-of-00002.gguf tier=cold rounds=3 arms=base new base=base$' "$OUT"
  count a-warmups "$OUT" '^WARMUP r0 (base|new) \| load_s ' 2
  count a-rows "$OUT" '^ROW r[123] (base|new) \| load_s ' 6
  count a-evictions "$OUT" '^    evict: /stub/model-00001-of-00002.gguf shards=2 pages=1000 resident before=900 after=0 \(0.00 %\)' 6
  check a-rotation "$(printf '%s\n' "$OUT" | grep -E '^ROW ' | awk '{printf "%s%s ", $2, $3}')" 'r1base r1new r2new r2base r3base r3new '
  matches a-fields '^ROW r1 base \| load_s 20.5 \| open_s 0.10 \| plan_s 0.30 \| context_s 0.20 \| upload_s 10.00 \| derive_s 0.40 \| host_set_s 1.50 \| body_s 1.00 \| head_s 0.10 \| other_s 0.20 \| wall_s [0-9.]+ \| stop_s [0-9.]+ \| evict_s [0-9.]+ resident_pct 0.00 \| majflt 0 \| pgpgin_gib 0.00 \| cached_gib 1.9->1.9 \| binary -$' "$OUT"
  matches a-record-lines '^    load phases open_s=0.10 ' "$OUT"
  matches a-upload-effect '^effect upload_s +new/base +means 10.00 -> 2.50 s +ratio 0.2500 +effect -75.0 % \+- 0.0 % \(n=3\)' "$OUT"
  matches a-load-effect '^effect load_s +new/base +means 20.50 -> 12.00 s +ratio 0.5854 ' "$OUT"
  matches a-done '^\[done\] sitting wall [0-9.]+ s, 8 starts, 0 failed$' "$OUT"
  # b: --warm evicts nothing, and its rows carry no eviction.
  sit STUB_ROUNDS=2 -- --warm base new -- --any
  check b-rc "$RC" 0
  count b-no-eviction "$OUT" '^    evict: ' 0
  matches b-config '^\[config\] .* tier=warm rounds=2 ' "$OUT"
  count b-rows "$OUT" '^ROW r[12] (base|new) \| .* evict_s - resident_pct - ' 4
  # c: 5 % resident after the drop (limit 1 %): every timed start fails by name and none runs, rc 1; the warm-ups ran.
  sit STUB_ROUNDS=2 STUB_RESIDENT=50 -- base new -- --any
  check c-rc "$RC" 1
  count c-fail-rows "$OUT" '^FAIL r[12] (base|new) rc=evict \| evict: 5.00 % of the 1000 pages of /stub/model-00001-of-00002.gguf.s 2 shard\(s\) stay resident after posix_fadvise\(DONTNEED\), over the 1 % the runner allows' 4
  count c-no-rows "$OUT" '^ROW ' 0
  count c-warmups-ran "$OUT" '^WARMUP r0 ' 2
  matches c-failed-list '^failed starts: r1 base rc=evict; r1 new rc=evict; r2 new rc=evict; r2 base rc=evict; $' "$OUT"
  # d: an advice that cannot run is its own failure.
  sit STUB_ROUNDS=2 STUB_NOADVICE=1 -- base new -- --any
  check d-rc "$RC" 1
  matches d-why "^FAIL r1 base rc=evict \| the advice over /stub/model-00001-of-00002.gguf's 2 shard\(s\) did not run: boom" "$OUT"
  # e: an arm that dies before its record fails its starts (rc and last line named); the pair with the base has no
  # round left, so no effect line, and the base's own rows stand.
  sit STUB_ROUNDS=2 -- base die -- --any
  check e-rc "$RC" 1
  # shellcheck disable=SC2016 # the backticks are the record's name in the message, not a substitution
  matches e-fail '^FAIL r1 die rc=3 \| generate_ds41 ended or ran out of its bound with no `load phases` record; last line: loading...' "$OUT"
  count e-base-rows "$OUT" '^ROW r[12] base \| ' 2
  count e-no-effect "$OUT" '^effect ' 0
  # f: refusals before anything runs.
  sit -- base -- --any
  check f-one-arm-rc "$RC" 64
  matches f-one-arm-line '^load-ab.sh: a base and one arm at least, got 1 remote dir\(s\)' "$OUT"
  sit -- --bin other base new
  check f-bin-args-rc "$RC" 64
  matches f-bin-args-line '^load-ab.sh: --bin other needs its arguments after --' "$OUT"
  sit -- --nope base new
  check f-flag-rc "$RC" 64
  echo "load-ab self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi

# The card pin, the stale-binary refusal and the other-card guard are the GPU runners' (timing-card.sh).
# shellcheck source=tools/ref/timing-card.sh
source "$HERE/timing-card.sh"
parse_args "$@"
check_bins
# shellcheck disable=SC2031 # the self-test's subshells set their own ROUNDS
T975=$(python3 "$HERE/tdist.py" "$ROUNDS") || exit 70
sitting
