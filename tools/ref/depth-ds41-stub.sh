#!/usr/bin/env bash
# The depth-ds41.sh stub test: the runner's arm loop with no lease, no card and no model. It copies the
# runner (DEPTH_DS41_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# timing-card.sh, cards.sh, lease-probe.sh, tdist.py, gguf-ranges.py, load-groups.sh, lcpp-fit.sh,
# cold-blocks.sh, lcpp-warm.sh and tools/bloomery, and a copy of lease.sh whose lease_take is replaced by a
# line that takes nothing; ref-paths.sh there is a stub
# profile whose engines are stub scripts (llama-bench, llama-server, generate_ds41, nvidia-smi) and whose
# MODEL is gguf-ranges.py's two-shard fixture. Nothing it starts loads a model or touches a card. The
# fault counter every copy reads (majflt_now, majflt_mark, lg_majflt, lcpp_srv_majflt) is a file the stub
# engines add to when a case makes them fault, so a row is cold exactly where a case says.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/depth-ds41-stub.sh'
#   ... 'DEPTH_DS41_RUNNER=/tmp/base-depth-ds41.sh bash tools/ref/depth-ds41-stub.sh'   # FAIL-first
#
# Runs on the box (bash 4 or later, GNU timeout, jq). One line per check, `ok <name>` or `FAIL <name>:
# <why>` followed by the run's output; exit 0 iff none failed. DEPTH_DS41_STUB_SHOW=1 prints every
# run's whole output after the checks.
#   failed-arm   6 lcpp:6 lcpp2:6 4 lcpppp:4 ik:6, two rounds: the stub llama-bench refuses
#                --n-cpu-moe 2 every time and the stub generate_ds41 refuses depth 4 once. The rows
#                of every other arm are there, the three failures are FAIL rows, the tables drop what
#                failed, and the runner ends at rc 1 naming them. The runner before FAIL rows stops at
#                the first failure: this check is red on it.
#   preheat      the same run's `preheat` lines carry the fixture's host-set bytes for K = 1 and 2, and
#                every row a majflt column (ours also the timed count).
#   corpus       code:4 code:4@STUB_X=1 lcpp:4: the code arms' rows and their own ratio table.
#   place        an ours arm whose SMOKE footer names another placement than --place is a FAIL row.
#   blocks       BLOOMERY_AB_ORDER=blocks, 6 lcpp:6 4 lcpppp:4 lcpp2:6 ik:6 code:4, two rounds: four blocks
#                (ours, lcpp, ik, code) in the order given, each opened by one DISCARD r0 row — the longest
#                prompt for ours and code, for lcpp the arm with the most token draws (lcpp:6, 13) at the
#                block's largest --n-cpu-moe (2), which the stub llama-bench echoes in its table — and each
#                block's rows rotated by one slot a round; the rows' heads are compared with that sequence
#                whole. The discards are in no mean; no preheat runs (its default under blocks is 0) and
#                no WARMUP row; every lcpp row carries a `timed` count from its --progress line and every
#                ik row a whole-process count (the stub ik llama-bench refuses --progress). Red on the
#                runner before blocks: it ignores the variable and rotates every arm.
#   order-bad    BLOOMERY_AB_ORDER=sideways is refused by name (rc 64).
#   blocks-dry   the same arms under BLOOMERY_DRY=1: the block plan, the lcpp discard's command line at
#                --n-cpu-moe 2 with --progress, and each block's rotation, then rc 0.
#   blocks-ph    BLOOMERY_AB_ORDER=blocks BLOOMERY_PREHEAT=1, lcpp:6 lcpp2:6, one round: the preheat runs
#                before every reference process, the discard's at the block's largest K (2).
#   grouped      6 4 code:4 lcpp:6, two rounds, no warm-up: the three generate_ds41 arms share one load a
#                round (one --arm list a process, 6 4 code:4 then 4 code:4 6: the units and the arms in
#                them rotated), each row between its own witness blocks and carrying its slot, the load's
#                lines once under [load], and the slot summary. Red on the runner before loads: a process
#                per arm, no --arm.
#   group-fail   6 4 5 with depth 4 refused once: the load's process ends at arm 4, a FAIL row, and 5 runs
#                in a fresh load.
#   load-arm     BLOOMERY_AB_LOAD=arm: every arm its own process.
#   solo         6 4 6@BLOOMERY_AB_LOAD=arm: the marked arm alone, under its own label and ratio.
#   grouped-dry  the dry run's loads per round and a grouped arm's command line.
#   fit          BLOOMERY_AB_ORDER=blocks, lcpp:6 lcppfit:6 lcppppfit:4 lcppppfit8:4, one round: the fit
#                arms are a block of their own (lcppfit), opened by the discard of lcppfit:6 (13 draws, no
#                --n-cpu-moe to raise); the stub llama-bench sees no -ngl or --n-cpu-moe from them, and
#                under -fitt -v prints two loads, the last with two overrides; every fit row, the discard's
#                included, carries the `fit` column read from that last load and echoes its lines, and the
#                lcpp row carries none. Red on the runner before the fit arms: arm usage, rc 64.
#   fit-preheat  lcppfit:6 under the default rotate order (preheat on): refused by name, rc 64 — the
#                preheat does not model the fit's host set.
#   fit-nobench  a llama-bench whose --help lists no --fit-target: the fit arms refused by name, rc 64.
#   fit-fail     a llama-bench whose fit fails (common_fit_params' warning) and runs anyway: each fit
#                arm is a FAIL row naming it, the discard's included, rc 1.
#   fit-dry      the dry run: a fit arm's command line (the profile's flags less -ngl and --n-cpu-moe,
#                then -fitt 1024 -v --progress) and what it dropped; the lcpp arm's line as before.
# The two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090, timing-card.sh; depth-qwen3moe-stub.sh has the
# pre-lease refusals), red on the runner before it:
#   twocard      lcpp:6 lcpppp:4, one round, the preheat on: the decode row and the json prefill row (its
#                device lines on stderr) carry `A6000+3090` and both cards, the stub llama-bench got -ts
#                1.5/1.5, the witness's two-card lines; rc 0.
#   twocard-arms 6, code:4, ik:6 and lcppsrv:6 each refused by name before anything runs (rc 64); ours under
#                --place a naming its two-card placement, bp, the server arm naming the two-card checks it lacks.
#   twocard-bp   BLOOMERY_GEN_PLACE=bp, lcpp:6 6, one round: ours runs (--place bp), its load record names
#                both cards, its row reads `A6000+3090` and `place bp`; rc 0.
#   twocard-bp-one  the same with a load record naming the A6000 alone (STUB_GEN_CARDS): ours is a FAIL row
#                naming the cards it saw; rc 1.
#   bp-onecard   BLOOMERY_GEN_PLACE=bp outside the two-card mode: refused by name, rc 64.
#   twocard-xid  lcpp:6 lcpppp:4, an Xid during the prefill arm: its FAIL row naming it, the decode row; rc 1.
#   twocard-dry  the dry run: the two-card lines, the precheck's `ok`, the lcpp line with -ts.
# The server arms and the warm rows (lcpp-warm.sh; red on the runner before them: arm usage, rc 64, or
# BLOOMERY_WARM_ROWS ignored):
#   srv          6 lcppsrv:6 4 lcppsrvpp:4 lcppsrvpp8:4, one round: each server arm's row (ids=lcg, its
#                warm-up, the continuation, W = prompt_ms + predicted_ms), the servers' command lines (the
#                profile's flags in its spellings, -fit off, -np 1 -ctxcp 0 --cache-ram 0, -c, port 0; the
#                decode arm's and the default prompt arm's alike, the lever's with -ub 8 -b 2048), the ids
#                each was sent twice (lease.sh's lcg_prompt), the ratio rows against ours, no server left up.
#   srv-blocks   BLOOMERY_AB_ORDER=blocks: the server arms are a block, its discard the longest prompt.
#   srv-exit, srv-hang, srv-badn  a server that exits before it listens (rc 5), one that never listens
#                (rc 124 after the arm bound), one whose timed answer has one predicted token too few:
#                each a FAIL row naming it, rc 1, no server left up.
#   srv-nobin, srv-probe  no llama-server (rc 2), a server whose --help lacks a flag the arm passes (rc
#                64): refused by name before the lease.
#   srv-corpus   code:4 lcppsrv:code:4: the server arm's row (ids=code, label lcppsrv@code) in the code
#                table, not in ours'.
#   srv-fit      lcppsrvfit:6: the server's fit flags and its fit column.
#   srv-dry      the dry run's server command line.
#   srv-cold     BLOOMERY_WARM_ROWS=1, a server whose timed request faults: a COLD row, then the retry's
#                row from one more request to the same server, rc 0.
#   srv-cold2    the retry faults too: the COLD row, then FAIL rc=cold, dropped, rc 1.
#   warm         BLOOMERY_WARM_ROWS=1, 6 4 lcpp:6, one round: ours' load runs 6 6 4 4 (each arm after its
#                PRIME); the ours arm at depth 4 faults twice (its prime and its row) and lcpp:6 once: a
#                COLD row each, then a fresh load of 4 4 and a second llama-bench, both rows clean, rc 0.
#   warm-fail    the same, cold again on the retry: FAIL rc=cold for ours at 4 and lcpp at 6, rc 1.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNNER=${DEPTH_DS41_RUNNER:-$ROOT/tools/ref/depth-ds41.sh}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/depth-ds41-stub.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
T=$tmp/tree
mkdir -p "$T/tools/ref" "$T/tools/bloomery" "$T/target/release" "$T/bin" "$T/data/engram" "$tmp/tmp"
cp "$RUNNER" "$T/tools/ref/depth-ds41.sh"
cp "$ROOT/tools/ref/timing-card.sh" "$ROOT/tools/ref/cards.sh" "$ROOT/tools/ref/lease-probe.sh" \
  "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" "$ROOT/tools/ref/gguf-ranges.py" \
  "$ROOT/tools/ref/load-groups.sh" "$ROOT/tools/ref/lcpp-fit.sh" "$ROOT/tools/ref/cold-blocks.sh" "$T/tools/ref/"
[ ! -f "$ROOT/tools/ref/lcpp-warm.sh" ] || cp "$ROOT/tools/ref/lcpp-warm.sh" "$T/tools/ref/"
cp -R "$ROOT/tools/bloomery/records.py" "$ROOT/tools/bloomery/schema" "$T/tools/bloomery/"
echo 'lease_take() { echo "[stub] no lease: the stub test'"'"'s copy of lease.sh takes nothing"; }' >> "$T/tools/ref/lease.sh"
# The stub's lease writes no record; it says it would write a two-card one (the precheck asks).
echo 'LEASE_CARDS_RECORD=1' >> "$T/tools/ref/lease.sh"
# The fault counter: $STUB_MAJFLT, a number the stub engines add to.
# shellcheck disable=SC2016 # the copies expand them when they run
{
  echo 'majflt_now() { cat "$STUB_MAJFLT"; }'
  echo 'majflt_mark() { awk -v f="$1" -v re="$2" -v src="$STUB_MAJFLT" '"'"'!s && re != "" && $0 ~ re { getline l < src; close(src); print l > f; close(f); s = 1 } { print; fflush() }'"'"'; }'
} >> "$T/tools/ref/cold-blocks.sh"
echo 'lg_majflt() { cat "$STUB_MAJFLT"; }' >> "$T/tools/ref/load-groups.sh"
[ ! -f "$T/tools/ref/lcpp-warm.sh" ] || echo 'lcpp_srv_majflt() { cat "$STUB_MAJFLT"; }' >> "$T/tools/ref/lcpp-warm.sh"
python3 "$T/tools/ref/gguf-ranges.py" fixture "$tmp/m-00001-of-00002.gguf" || exit 2
seq 101 110 > "$T/data/engram/corpus-code.ids"
cat > "$T/tools/ref/ref-paths.sh" << EOF
# shellcheck shell=bash
MODEL_NAME=deepseek41
MODEL=$tmp/m-00001-of-00002.gguf
IK=$T IKBIN=$T/bin/ik-bench LCPP=$T LCPPBIN=$T/bin/lcpp-bench
IK_GPU_FLAGS='-ngl 999 --n-cpu-moe 1 -t 4' IK_GPU_ENV=''
LCPP_GPU_FLAGS='-ngl 999 --n-cpu-moe 1 -fa on -t 4'
BLOOMERY_DATA=$T/data
TWO_CARD_PLACEMENT=
if [ "\${BLOOMERY_TIMING_CARDS:-}" = a6000+3090 ]; then
  LCPP_GPU_FLAGS='-ngl 999 --n-cpu-moe 1 -fa on -t 4 -ts 1.5/1.5'
  TWO_CARD_PLACEMENT='the stub two-card line: --n-cpu-moe 1 -ts 1.5/1.5'
fi
EOF
# shellcheck source=tools/ref/depth-stub-cards.sh
. "$HERE/depth-stub-cards.sh"
# The stub llama-bench: refuses --n-cpu-moe STUB_BENCH_FAIL_K; a pp run (-o json) prints one test with
# two samples; a decode run prints the markdown row under the engine's label, its model column naming
# the --n-cpu-moe it ran at. Under --progress it prints mainline's progress lines on stderr (the ik
# copy refuses the flag, as ik's llama-bench does), the error of a refused K last. --help lists
# --fit-target unless STUB_BENCH_NO_FIT is set; under -fitt it prints common_fit_params' failure warning
# when STUB_BENCH_FIT_FAIL is set, and under -fitt -v two model loads on stderr, the fit's measuring one
# and the real one with two expert tensors of blk 1 overridden to the host.
cat > "$T/bin/bench" << 'EOF'
#!/usr/bin/env bash
eng=${0##*/} k='' p='' n='' d='' gp='' json='' prog='' fitt='' verb='' ts=''
while [ $# -gt 0 ]; do
  case $1 in
    --n-cpu-moe) k=$2; shift ;; -p) p=$2; shift ;; -n) n=$2; shift ;; -d) d=$2; shift ;; -gp) gp=$2; shift ;;
    -o) json=1; shift ;; --progress) prog=1 ;; -fitt) fitt=$2; shift ;; -v) verb=1 ;; -ts) ts=$2; shift ;;
    -h | --help)
      echo "usage: $eng [options]"
      [ -n "${STUB_BENCH_NO_FIT:-}" ] || echo "  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)"
      exit 0
      ;;
  esac
  shift
done
if [ -n "$fitt" ]; then
  [ -z "${STUB_BENCH_FIT_FAIL:-}" ] || echo "common_fit_params: failed to fit params to free device memory: stub" >&2
  if [ -n "$verb" ]; then
    for l in "llama_model_loader: loaded meta data with 3 key-value pairs and 6 tensors from stub" \
      "load_tensors: offloaded 3/3 layers to GPU" "load_tensors:        CUDA0 model buffer size =    12.00 MiB" \
      "llama_model_loader: loaded meta data with 3 key-value pairs and 6 tensors from stub" \
      "tensor blk.1.ffn_up_exps.weight (1 MiB q4_K) buffer type overridden to CPU" \
      "tensor blk.1.ffn_down_exps.weight (1 MiB q4_K) buffer type overridden to CPU" \
      "load_tensors: offloaded 3/3 layers to GPU" "load_tensors:        CUDA0 model buffer size =    10.00 MiB" \
      "load_tensors:   CPU_Mapped model buffer size =     2.00 MiB"; do
      echo "$l" >&2
    done
  fi
fi
if [ -n "$prog" ] && [ "$eng" = ik-bench ]; then
  echo "error: unknown argument: --progress" >&2
  exit 1
fi
if [ -n "$p" ] && [ "$p" != 0 ]; then label="pp$p"; elif [ "$eng" = ik-bench ]; then label="tg${gp#*,}@pp${gp%,*}"; else label="tg$n @ d$d"; fi
. "${STUB_BENCH_CARDS:?}"
[ -z "$prog" ] || echo "llama-bench: benchmark 1/1: starting" >&2
if [ "$k" = "${STUB_BENCH_FAIL_K:-none}" ]; then
  echo "llama_init_from_model: failed to create context (stub: --n-cpu-moe $k)" >&2
  exit 1
fi
# fault_once <key>: the first K runs of the test <key> (d<D> or p<P>) that STUB_COLD_BENCH=<key>:<K> names
# add 100000 faults to the counter, 0.3 s after the timed window's progress line.
fault_once() {
  local m=${TMPDIR:-/tmp}/stub-cold-bench-$1 c
  [ "${STUB_COLD_BENCH%%:*}" = "$1" ] || return 0
  c=$(cat "$m" 2> /dev/null || echo 0)
  [ "$c" -lt "${STUB_COLD_BENCH#*:}" ] || return 0
  echo $((c + 1)) > "$m"
  sleep 0.3
  echo $(($(cat "$STUB_MAJFLT") + 100000)) > "$STUB_MAJFLT"
}
if [ -n "$json" ]; then
  if [ -n "$prog" ]; then
    for l in "warmup prompt run" "prompt run 1/2" "prompt run 2/2"; do echo "llama-bench: benchmark 1/1: $l" >&2; done
  fi
  fault_once "p$p"
  echo "[{\"n_prompt\": $p, \"samples_ns\": [2000000000, 1000000000], \"build_commit\": \"stub\", \"build_number\": 0, \"gpu_info\": \"stub card\"${ts:+, \"tensor_split\": \"$ts\"}}]"
  exit 0
fi
if [ -n "$prog" ]; then
  echo "llama-bench: benchmark 1/1: warmup generation run" >&2
  [ -z "$d" ] || echo "llama-bench: benchmark 1/1: depth run 1/1" >&2
  echo "llama-bench: benchmark 1/1: generation run 1/1" >&2
fi
fault_once "d$d"
if [ "$eng" = ik-bench ]; then label="tg${gp#*,}@pp${gp%,*}"; else label="tg$n @ d$d"; fi
echo "| model | size | test | t/s |"
echo "| --- | ---: | ---: | ---: |"
echo "| stub k=$k${ts:+ ts=$ts} | 1 | $label | 20.00 ± 0.01 |"
echo "build: stub (0)"
EOF
cp "$T/bin/bench" "$T/bin/ik-bench"
mv "$T/bin/bench" "$T/bin/lcpp-bench"
touch "$T/Cargo.toml"
# The stub generate_ds41: refuses depth STUB_GEN_FAIL_DEPTH the first time (a marker file), and names
# STUB_GEN_PLACE in its SMOKE footer when set. Under --place bp, or with STUB_GEN_CARDS, its load line
# names the cards (`cards=`, both under bp unless STUB_GEN_CARDS says otherwise). Under --arm it runs the list after one `load` line, each
# arm opening with its `arm` record and, under --arm-sync, waiting for a line on stdin; every process
# appends a line to $TMPDIR/stub-gen-loads.
cat > "$T/target/release/generate_ds41" << 'EOF'
#!/usr/bin/env bash
depth='' n=32 place=a tokens='' sync='' arms=()
while [ $# -gt 0 ]; do
  case $1 in
    --depth) depth=$2; shift ;; --tokens) tokens=$2; shift ;; -n) n=$2; shift ;; --place) place=$2; shift ;;
    --warm) shift ;; --arm) arms+=("$2"); shift ;; --arm-sync) sync=1 ;;
  esac
  shift
done
echo "${arms[*]:-one}" >> "${TMPDIR:-/tmp}/stub-gen-loads"
[ -z "$tokens" ] || depth=$(echo "$tokens" | tr ',' '\n' | grep -c .)
[ ${#arms[@]} -gt 0 ] || arms=("$depth")
echo "plan place=$place (stub)"
cards=${STUB_GEN_CARDS:-}
[ -n "$cards" ] || [ "$place" != bp ] || cards='[A6000,3090]'
if [ -n "$cards" ]; then
  echo "load resident_bytes=0 place=$place cards=$cards arms=${#arms[@]} (stub)"
else
  echo "load place=$place arms=${#arms[@]} (stub)"
fi
for k in "${!arms[@]}"; do
  a=${arms[$k]} feed=lcg
  case $a in *:*) feed=${a%%:*} depth=${a#*:} ;; *) depth=$a ;; esac
  if [ -n "$sync" ]; then
    echo "arm i=$k arms=${#arms[@]} feed=$feed ids=$depth n=$n"
    read -r _ || { echo "error: stdin closed before arm $k" >&2; exit 65; }
  fi
  mark=${TMPDIR:-/tmp}/stub-gen-failed-$depth
  if [ "$depth" = "${STUB_GEN_FAIL_DEPTH:-none}" ] && [ ! -e "$mark" ]; then
    touch "$mark"
    echo "error: the stub refuses depth $depth once" >&2
    exit 3
  fi
  echo "fed ids=$depth first=[1,2,3,4] last=[5,6,7,8] depth_sequence_from=0"
  # STUB_COLD_GEN=<depth>:<K>: the first K arms of that depth add 100000 faults after their fed line.
  cm=${TMPDIR:-/tmp}/stub-cold-gen-$depth
  if [ "${STUB_COLD_GEN%%:*}" = "$depth" ] && [ "$(cat "$cm" 2> /dev/null || echo 0)" -lt "${STUB_COLD_GEN#*:}" ]; then
    echo $(($(cat "$cm" 2> /dev/null || echo 0) + 1)) > "$cm"
    sleep 0.3
    echo $(($(cat "$STUB_MAJFLT") + 100000)) > "$STUB_MAJFLT"
  fi
  echo "time prompt n=$depth ms=100.0000 tok/s=$((depth * 10)).00 passes=1 kind=batch"
  for i in $(seq 0 $((n - 1))); do echo "step $i $((depth + i)) $((1000 + i))"; done
  for i in $(seq 1 $((n - 1))); do echo "time step $i ms=33.0000"; done
  echo "SMOKE mode=graph place=${STUB_GEN_PLACE:-$place} prompt_tokens=0 depth=$depth generated=$n warm=0 steps=$((n - 1)) p50_ms=33.0000 mean_ms=33.0000 tok/s(p50)=30.30"
done
EOF
# The stub llama-server: tools/ref/stub-llama-server.py (its docstring has what it answers and the cases).
cp "$ROOT/tools/ref/stub-llama-server.py" "$T/bin/llama-server"
chmod +x "$T/bin/"* "$T/target/release/generate_ds41"

n=0 failed=0
pass() { n=$((n + 1)); echo "ok $1"; }
fail() {
  n=$((n + 1)) failed=$((failed + 1))
  echo "FAIL $1: $2"
  [ -z "${3:-}" ] || sed 's/^/    | /' "$3"
}
# stub_run <log> <env…> -- <arms…>: the runner in the stub tree; its rc into RC.
stub_run() {
  local log=$1 e=()
  shift
  while [ "$1" != -- ]; do e+=("$1"); shift; done
  shift
  rm -f "$tmp/tmp"/stub-gen-failed-* "$tmp/tmp/stub-gen-loads" "$tmp/tmp"/stub-once-* "$tmp/tmp/stub-xid" \
    "$tmp/tmp"/stub-cold-* "$tmp/tmp"/stub-srv-*
  echo 0 > "$tmp/tmp/stub-majflt"
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 \
    BLOOMERY_CPU_BUSY_COMMS=none STUB_BENCH_CARDS="$T/bin/stub-bench-cards" TIMING_CARDS_POLL=1 \
    STUB_MAJFLT="$tmp/tmp/stub-majflt" "${e[@]}" bash tools/ref/depth-ds41.sh "$@") > "$log" 2>&1
  RC=$?
}
# srv_left: the stub servers of the last run still up (pids from their own file), none when all stopped.
srv_left() {
  local p left=''
  [ -f "$tmp/tmp/stub-srv-pids" ] || return 0
  while read -r p; do ! kill -0 "$p" 2> /dev/null || left+="$p "; done < "$tmp/tmp/stub-srv-pids"
  echo "$left"
}
# want <name> <log> <count> <pattern>: the log holds exactly <count> lines matching grep -E <pattern>.
want() {
  local c
  c=$(grep -cE -- "$4" "$2")
  [ "$c" = "$3" ] || { fail "$1" "$c lines match /$4/, want $3" "$2"; return 1; }
}

L=$tmp/failed-arm.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 STUB_BENCH_FAIL_K=2 STUB_GEN_FAIL_DEPTH=4 -- 6 lcpp:6 lcpp2:6 4 lcpppp:4 ik:6
if [ "$RC" != 1 ]; then
  fail failed-arm "rc $RC, want 1" "$L"
elif want failed-arm "$L" 2 '^FAIL r[12] lcpp2 d=6 rc=1 \| no .tg4 @ d6. row; last line: llama_init_from_model: failed to create context' &&
  want failed-arm "$L" 1 '^FAIL r1 ours d=4 rc=3 \| exited 3' &&
  want failed-arm "$L" 9 '^ROW ' &&
  want failed-arm "$L" 1 '^WARMUP r0 ours d=6 ' &&
  want failed-arm "$L" 2 '^ROW r2 (ours d=4|lcpppp p=4) ' &&
  want failed-arm "$L" 1 '^    dropped: lcpp2 at 6$' &&
  want failed-arm "$L" 1 '^    dropped: ours at 4$' &&
  want failed-arm "$L" 0 '^mean (pp )?(lcpp2|ours) (d|p)=4|^mean lcpp2 ' &&
  want failed-arm "$L" 1 '^ratio d=6 +ours/lcpp ' &&
  want failed-arm "$L" 1 '^failed arms: r1 ours d=4 rc=3; r1 lcpp2 d=6 rc=1; r2 lcpp2 d=6 rc=1; $'; then
  pass failed-arm
fi
if [ "$RC" = 1 ] && grep -q '^failed arms: ' "$L"; then
  if want preheat "$L" 6 '^preheat (lcpp|lcpppp|ik) K=1 bytes=1416 s=[0-9.]+ gbps=' &&
    want preheat "$L" 2 '^preheat lcpp2 K=2 bytes=2136 ' &&
    want preheat "$L" 9 '^ROW .* \| majflt [0-9]+ \(' &&
    want preheat "$L" 3 '^ROW .* ours d=[46] .*\| majflt [0-9]+ \(timed [0-9]+; ≤ [0-9.]+ % of W ' &&
    want preheat "$L" 1 '^\[config\] preheat: host K=1 layers=2 tensors=3 bytes=1416 '; then
    pass preheat
  fi
else
  fail preheat "the failed-arm run did not reach its end (rc $RC)" "$L"
fi

L=$tmp/corpus.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 -- code:4 code:4@STUB_X=1 lcpp:4
if [ "$RC" != 0 ]; then
  fail corpus "rc $RC, want 0" "$L"
elif want corpus "$L" 2 '^ROW r[12] code d=4 n=4 .* \| place a \|' &&
  want corpus "$L" 2 '^ROW r[12] code@STUB_X=1 d=4 ' &&
  want corpus "$L" 1 '^=== the code prompt: code / each code@ arm per P' &&
  want corpus "$L" 1 '^ratio code d=4 +code/code@STUB_X=1 ' &&
  want corpus "$L" 0 '^ratio d=4 +ours/code' &&
  want corpus "$L" 1 '^\[config\] code: the first P ids of .*/corpus-code.ids \(10 ids\)$'; then
  pass corpus
fi

L=$tmp/place.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_GEN_PLACE=gate -- 6
if [ "$RC" != 1 ]; then
  fail place "rc $RC, want 1" "$L"
elif want place "$L" 1 '^FAIL r1 ours d=6 rc=0 \| its SMOKE footer names place=gate; the runner passed --place a' &&
  want place "$L" 0 '^ROW '; then
  pass place
fi

BLOCK_ARMS=(6 lcpp:6 4 lcpppp:4 lcpp2:6 ik:6 code:4)
L=$tmp/blocks.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_ORDER=blocks -- "${BLOCK_ARMS[@]}"
want_seq="DISCARD r0 ours d=6
ROW r1 ours d=6
ROW r1 ours d=4
ROW r2 ours d=4
ROW r2 ours d=6
DISCARD r0 lcpp d=6
ROW r1 lcpp d=6
ROW r1 lcpppp p=4
ROW r1 lcpp2 d=6
ROW r2 lcpppp p=4
ROW r2 lcpp2 d=6
ROW r2 lcpp d=6
DISCARD r0 ik d=6
ROW r1 ik d=6
ROW r2 ik d=6
DISCARD r0 code d=4
ROW r1 code d=4
ROW r2 code d=4"
seq=$(grep -oE '^(DISCARD|WARMUP|ROW|FAIL) r[0-9]+ [^ ]+ [dp]=[0-9]+' "$L")
if [ "$RC" != 0 ]; then
  fail blocks "rc $RC, want 0" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail blocks "the rows' heads are not the block order: $(echo "$seq" | paste -sd'|' -)" "$L"
elif want blocks "$L" 1 '^\[config\] order: blocks, a discard before each block: on$' &&
  want blocks "$L" 1 '^\[config\] block 2/4 lcpp: lcpp:6 lcpppp:4 lcpp2:6; discard lcpp:6 at --n-cpu-moe 2, the most token draws of the block.s arms \(lcpp:6 13, lcpppp:4 12, lcpp2:6 13\), at the block.s largest --n-cpu-moe$' &&
  want blocks "$L" 1 '^\[config\] block 1/4 ours: 6 4; discard 6, the longest prompt' &&
  want blocks "$L" 4 '^\[block\] [1-4]/4 ' &&
  want blocks "$L" 3 '^    lcpp2? table \| stub k=2 ' &&
  want blocks "$L" 2 '^    lcpp table \| stub k=1 ' &&
  want blocks "$L" 1 '^mean ours d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^mean lcpp d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^mean ik d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^\[config\] preheat: off \(BLOOMERY_AB_ORDER=blocks' &&
  want blocks "$L" 0 '^preheat |^WARMUP ' &&
  want blocks "$L" 7 '^(DISCARD|ROW) r[0-2] lcpp(2|pp)? .*\| majflt [0-9]+ \(timed [0-9]+; ≤ ' &&
  want blocks "$L" 3 '^(DISCARD|ROW) r[0-2] ik .*\| majflt [0-9]+ \(whole process; ≤ ' &&
  want blocks "$L" 1 '^\[config\] cold tag: .* × 75 µs ≥ 1 % ' &&
  want blocks "$L" 1 '^failed arms: 0 '; then
  pass blocks
fi

L=$tmp/order-bad.log
stub_run "$L" BLOOMERY_AB_ORDER=sideways -- 6
if [ "$RC" != 64 ]; then
  fail order-bad "rc $RC, want 64" "$L"
elif want order-bad "$L" 1 "^depth-ds41.sh: BLOOMERY_AB_ORDER is rotate .* or blocks .*, got 'sideways'$"; then
  pass order-bad
fi

L=$tmp/blocks-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_ORDER=blocks BLOOMERY_DRY=1 -- "${BLOCK_ARMS[@]}"
if [ "$RC" != 0 ]; then
  fail blocks-dry "rc $RC, want 0" "$L"
elif want blocks-dry "$L" 1 '^\[dry\] block 2/4 lcpp: lcpp:6 lcpppp:4 lcpp2:6; discard lcpp:6 at --n-cpu-moe 2,' &&
  want blocks-dry "$L" 1 '^\[dry\] block 2 discard lcpp:6: timeout .*lcpp-bench -m .* -p 0 -n 4 -d 6 -r 1 -ngl 999 -fa on -t 4 --n-cpu-moe 2 --progress ' &&
  want blocks-dry "$L" 1 '^\[dry\] block 2 round 2 order: lcpppp:4 lcpp2:6 lcpp:6$' &&
  want blocks-dry "$L" 1 '^\[dry\] block 1 round 1 order: 6 4$' &&
  want blocks-dry "$L" 1 '^\[dry\] block 4 discard code:4: ' &&
  want blocks-dry "$L" 1 '^\[dry\] warmup: the blocks. discards below' &&
  want blocks-dry "$L" 0 '^\[dry\] round '; then
  pass blocks-dry
fi

L=$tmp/blocks-ph.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks BLOOMERY_PREHEAT=1 -- lcpp:6 lcpp2:6
if [ "$RC" != 0 ]; then
  fail blocks-ph "rc $RC, want 0" "$L"
elif want blocks-ph "$L" 3 '^preheat ' &&
  want blocks-ph "$L" 1 '^preheat lcpp K=2 bytes=2136 ' &&
  want blocks-ph "$L" 1 '^preheat lcpp K=1 bytes=1416 ' &&
  want blocks-ph "$L" 1 '^preheat lcpp2 K=2 bytes=2136 ' &&
  want blocks-ph "$L" 1 '^DISCARD r0 lcpp d=6 '; then
  pass blocks-ph
fi

LOADS=$tmp/tmp/stub-gen-loads
L=$tmp/grouped.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 -- 6 4 code:4 lcpp:6
want_seq="ROW r1 ours d=6
ROW r1 ours d=4
ROW r1 code d=4
ROW r1 lcpp d=6
ROW r2 lcpp d=6
ROW r2 ours d=4
ROW r2 code d=4
ROW r2 ours d=6"
seq=$(grep -oE '^(DISCARD|WARMUP|ROW|FAIL) r[0-9]+ [^ ]+ [dp]=[0-9]+' "$L")
if [ "$RC" != 0 ]; then
  fail grouped "rc $RC, want 0" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail grouped "the rows' heads are not the load order: $(echo "$seq" | paste -sd'|' -)" "$L"
elif [ "$(paste -sd'|' - < "$LOADS")" != "6 4 code:4|4 code:4 6" ]; then
  fail grouped "the processes' --arm lists: $(paste -sd'|' - < "$LOADS"), want 6 4 code:4|4 code:4 6" "$L"
elif want grouped "$L" 2 '^\[load\] r[12] 3 arm\(s\): ' &&
  want grouped "$L" 2 '^    load place=a arms=3 \(stub\)$' &&
  want grouped "$L" 1 '^ROW r1 ours d=6 .*\| majflt [0-9]+ \(timed [0-9]+; .*\| slot 1/3 \| wall ' &&
  want grouped "$L" 1 '^ROW r2 ours d=6 .*\| slot 3/3 \| wall ' &&
  want grouped "$L" 1 '^ROW r1 code d=4 .*\| slot 3/3 \| wall ' &&
  want grouped "$L" 1 '^ROW r2 code d=4 .*\| slot 2/3 \| wall ' &&
  want grouped "$L" 0 '^ROW r[12] lcpp .*\| slot ' &&
  want grouped "$L" 1 '^slots ours p=6 .*first: pp 60.00 tg 30.30 \(n=1\)  later: pp 60.00 tg 30.30 \(n=1\)  later/first pp 1.0000$' &&
  want grouped "$L" 12 '^--- witness (pre|post) r[12] (ours|code) '; then
  pass grouped
fi

L=$tmp/group-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_GEN_FAIL_DEPTH=4 -- 6 4 5
if [ "$RC" != 1 ]; then
  fail group-fail "rc $RC, want 1" "$L"
elif [ "$(paste -sd'|' - < "$LOADS")" != "6 4 5|5" ]; then
  fail group-fail "the processes' --arm lists: $(paste -sd'|' - < "$LOADS"), want 6 4 5|5" "$L"
elif want group-fail "$L" 1 '^ROW r1 ours d=6 .*\| slot 1/3 \|' &&
  want group-fail "$L" 1 '^FAIL r1 ours d=4 rc=3 \| exited 3; last line: error: the stub refuses depth 4 once' &&
  want group-fail "$L" 1 '^\[load\] r1: arm 4 failed \(rc 3\); the 1 arm\(s\) after it run in a fresh load$' &&
  want group-fail "$L" 1 '^ROW r1 ours d=5 .*\| slot 1/1 \|' &&
  want group-fail "$L" 1 '^failed arms: r1 ours d=4 rc=3; $'; then
  pass group-fail
fi

L=$tmp/load-arm.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_AB_LOAD=arm -- 6 4
if [ "$RC" != 0 ]; then
  fail load-arm "rc $RC, want 0" "$L"
elif [ "$(paste -sd'|' - < "$LOADS")" != "6|4" ]; then
  fail load-arm "the processes' --arm lists: $(paste -sd'|' - < "$LOADS"), want 6|4" "$L"
elif want load-arm "$L" 2 '^ROW r1 ours d=[46] .*\| slot 1/1 \|'; then
  pass load-arm
fi

L=$tmp/solo.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 -- 6 4 6@BLOOMERY_AB_LOAD=arm
if [ "$RC" != 0 ]; then
  fail solo "rc $RC, want 0" "$L"
elif [ "$(paste -sd'|' - < "$LOADS")" != "6 4|6" ]; then
  fail solo "the processes' --arm lists: $(paste -sd'|' - < "$LOADS"), want 6 4|6" "$L"
elif want solo "$L" 1 '^ROW r1 ours@BLOOMERY_AB_LOAD=arm d=6 .*\| slot 1/1 \|' &&
  want solo "$L" 1 '^ratio d=6 +ours/ours@BLOOMERY_AB_LOAD=arm '; then
  pass solo
fi

L=$tmp/grouped-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- 6 4 lcpp:6
if [ "$RC" != 0 ]; then
  fail grouped-dry "rc $RC, want 0" "$L"
elif want grouped-dry "$L" 1 '^\[dry\] round 1 loads: \[6 4\] lcpp:6$' &&
  want grouped-dry "$L" 1 '^\[dry\] round 2 loads: lcpp:6 \[4 6\]$' &&
  want grouped-dry "$L" 1 '^\[dry\] 4: one arm of a load: .*generate_ds41 --arm 4 -n 4 --place a --time --arm-sync '; then
  pass grouped-dry
fi

FIT_ROW='\| fit offloaded 3/3, overridden CPU:2 in blk 1-1 \(blk 1: 2\), buffers CUDA0 10.00 CPU_Mapped 2.00 MiB \|'
L=$tmp/fit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks -- lcpp:6 lcppfit:6 lcppppfit:4 lcppppfit8:4
if [ "$RC" != 0 ]; then
  fail fit "rc $RC, want 0" "$L"
elif want fit "$L" 1 "^\[config\] block 2/2 lcppfit: lcppfit:6 lcppppfit:4 lcppppfit8:4; discard lcppfit:6, the most token draws of the block.s arms \(lcppfit:6 13, lcppppfit:4 12, lcppppfit8:4 12\)$" &&
  want fit "$L" 1 "^\[config\] lcppfit: .*/lcpp-bench flags=-fa on -t 4 -fitt 1024 -v \(llama-bench.s fit places the model; dropped: -ngl 999 --n-cpu-moe 1;" &&
  want fit "$L" 1 "^DISCARD r0 lcppfit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) $FIT_ROW build " &&
  want fit "$L" 1 "^ROW r1 lcppfit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) $FIT_ROW build " &&
  want fit "$L" 1 "^ROW r1 lcppppfit p=4 n=0 \| tok/s\(pp\) [0-9.]+ @ n=0, prompt 4, A6000 \(stub\) $FIT_ROW cold .* \| ub 512 b 2048 \(llama-bench defaults\) \|" &&
  want fit "$L" 1 "^ROW r1 lcppppfit8 p=4 n=0 .*$FIT_ROW cold .* \| ub 8 b 2048 \(the arm.s lever\) \|" &&
  want fit "$L" 1 '^ROW r1 lcpp d=6 ' &&
  want fit "$L" 0 '^(ROW|DISCARD) r[01] lcpp (d|p)=.*\| fit ' &&
  want fit "$L" 4 '^    lcpp(pp)?fit8? fit load_tensors: offloaded 3/3 layers to GPU$' &&
  want fit "$L" 4 '^    lcpp(pp)?fit8? fit load_tensors: +CPU_Mapped model buffer size = +2.00 MiB$' &&
  want fit "$L" 2 '^    lcppfit table \| stub k= \|' &&
  want fit "$L" 1 '^mean lcppfit d=6 ' &&
  want fit "$L" 1 '^failed arms: 0 '; then
  pass fit
fi

L=$tmp/fit-preheat.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- lcppfit:6
if [ "$RC" != 64 ]; then
  fail fit-preheat "rc $RC, want 64" "$L"
elif want fit-preheat "$L" 1 "^depth-ds41.sh: arm 'lcppfit:6': its flags carry -fitt, a host set the preheat does not model; set BLOOMERY_PREHEAT=0" &&
  want fit-preheat "$L" 0 '^(ROW|WARMUP|preheat) '; then
  pass fit-preheat
fi

L=$tmp/fit-nobench.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks STUB_BENCH_NO_FIT=1 -- lcpp:6 lcppfit:6
if [ "$RC" != 64 ]; then
  fail fit-nobench "rc $RC, want 64" "$L"
elif want fit-nobench "$L" 1 "^depth-ds41.sh: the lcppfit/lcppppfit arms need llama-bench.s fit: .*/lcpp-bench --help lists no -fitt/--fit-target: this llama-bench has no fit \(tree " &&
  want fit-nobench "$L" 0 '^(ROW|DISCARD) '; then
  pass fit-nobench
fi

L=$tmp/fit-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks STUB_BENCH_FIT_FAIL=1 -- lcppfit:6 lcppppfit:4
if [ "$RC" != 1 ]; then
  fail fit-fail "rc $RC, want 1" "$L"
elif want fit-fail "$L" 2 "^FAIL r[01] lcppfit d=6 rc=0 \| llama-bench.s fit failed, and llama-bench loads without it: common_fit_params: failed to fit params to free device memory: stub" &&
  want fit-fail "$L" 1 "^FAIL r1 lcppppfit p=4 rc=0 \| llama-bench.s fit failed, " &&
  want fit-fail "$L" 0 '^(ROW|DISCARD) '; then
  pass fit-fail
fi

L=$tmp/fit-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks BLOOMERY_DRY=1 -- lcpp:6 lcppfit:6 lcppppfit:4
if [ "$RC" != 0 ]; then
  fail fit-dry "rc $RC, want 0" "$L"
elif want fit-dry "$L" 1 "^\[dry\] lcppfit:6: timeout --kill-after=10 60 env  [^ ]*/lcpp-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -fa on -t 4 -fitt 1024 -v --progress   # row label 'tg4 @ d6', measured window from /: generation run 1/1\\\$/, placement: llama-bench.s fit at -fitt 1024 MiB \(dropped: -ngl 999 --n-cpu-moe 1\), -v for the fit column$" &&
  want fit-dry "$L" 1 "^\[dry\] lcppppfit:4: timeout .*/lcpp-bench -m [^ ]* -p 4 -n 0 -r 2 -o json -fa on -t 4 -fitt 1024 -v --progress   # row label 'pp4', ub 512 b 2048 \(llama-bench defaults\), measured window from /: prompt run 2/2\\\$/, placement: " &&
  want fit-dry "$L" 1 "^\[dry\] lcpp:6: timeout .*/lcpp-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -ngl 999 --n-cpu-moe 1 -fa on -t 4 --progress   # row label 'tg4 @ d6', measured window from /: generation run 1/1\\\$/$" &&
  want fit-dry "$L" 1 '^\[dry\] block 2/2 lcppfit: lcppfit:6 lcppppfit:4; discard lcppfit:6, '; then
  pass fit-dry
fi

# The two-card mode (red on the runner before it).
TC=BLOOMERY_TIMING_CARDS=a6000+3090
L=$tmp/twocard.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" -- lcpp:6 lcpppp:4
if [ "$RC" != 0 ]; then
  fail twocard "rc $RC, want 0" "$L"
elif want twocard "$L" 1 '^ROW r1 lcpp d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 \| build stub \(0\) \| device NVIDIA RTX A6000 \(stub\) \+ NVIDIA GeForce RTX 3090 \(stub\) \| majflt ' &&
  want twocard "$L" 1 '^ROW r1 lcpppp p=4 n=0 \| tok/s\(pp\) 4 @ n=0, prompt 4, A6000\+3090 \| cold 2 \(repetition 1 of 2\) \| ub 512 b 2048 \(llama-bench defaults\) \| build stub \(0\) \| device stub card \| majflt ' &&
  want twocard "$L" 1 '^    lcpp table \| stub k=1 ts=1.5/1.5 \| ' &&
  want twocard "$L" 1 '^    lcpppp params .*"tensor_split":"1.5/1.5"' &&
  want twocard "$L" 2 '^preheat lcpp(pp)? K=1 ' &&
  want twocard "$L" 0 '^    timing-card: ' &&
  want twocard "$L" 6 '^    3090 cap: ok ' &&
  want twocard "$L" 6 '^    xid: 0 NVRM Xid line\(s\) since the lease was taken ' &&
  want twocard "$L" 1 '^\[config\] two cards: A6000\+3090, the profile.s two-card line: the stub two-card line: '; then
  pass twocard
fi

# twocard_refused <name> <pattern> <arms…>: the two-card run of those arms refused, rc 64, the pattern on
# one line, and no row, witness or lease.
twocard_refused() {
  local name=$1 pat=$2
  shift 2
  L=$tmp/$name.log
  stub_run "$L" "$TC" -- "$@"
  if [ "$RC" != 64 ]; then
    fail "$name" "rc $RC, want 64" "$L"
  elif want "$name" "$L" 1 "$pat" && want "$name" "$L" 0 '^(ROW|FAIL|DISCARD|WARMUP) |^--- witness|^\[stub\] no lease'; then
    pass "$name"
  fi
}
twocard_refused twocard-arms "^depth-ds41.sh: arm '6': generate_ds41 under --place a loads one card; its two-card placement is --place bp \(BLOOMERY_GEN_PLACE=bp\)$" lcpp:6 6
twocard_refused twocard-arms-code "^depth-ds41.sh: arm 'code:4': generate_ds41 under --place a loads one card; " lcpp:6 code:4
twocard_refused twocard-arms-ik "^depth-ds41.sh: arm 'ik:6': the two-card table.s reference is mainline llama.cpp .*; ik has no two-card arm$" lcpp:6 ik:6
twocard_refused twocard-arms-srv "^depth-ds41.sh: arm 'lcppsrv:6': a llama-server arm has no two-card checks yet " lcpp:6 lcppsrv:6

L=$tmp/twocard-bp.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" BLOOMERY_GEN_PLACE=bp -- lcpp:6 6
if [ "$RC" != 0 ]; then
  fail twocard-bp "rc $RC, want 0" "$L"
elif want twocard-bp "$L" 1 '^ROW r1 ours d=6 n=4 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place bp \| ' &&
  want twocard-bp "$L" 1 '^ROW r1 lcpp d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 ' &&
  want twocard-bp "$L" 1 '^    load resident_bytes=0 place=bp cards=\[A6000,3090\] ' &&
  want twocard-bp "$L" 0 '^FAIL '; then
  pass twocard-bp
fi

L=$tmp/twocard-bp-one.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" BLOOMERY_GEN_PLACE=bp 'STUB_GEN_CARDS=[A6000]' -- 6
if [ "$RC" != 1 ]; then
  fail twocard-bp-one "rc $RC, want 1" "$L"
elif want twocard-bp-one "$L" 1 "^FAIL r1 ours d=6 rc=0 \| two cards: the engine.s load record names cards \[A6000\], not the A6000 and the 3090: a one-card run in the two-card table" &&
  want twocard-bp-one "$L" 0 '^ROW '; then
  pass twocard-bp-one
fi

L=$tmp/bp-onecard.log
stub_run "$L" BLOOMERY_GEN_PLACE=bp -- 6
if [ "$RC" != 64 ]; then
  fail bp-onecard "rc $RC, want 64" "$L"
elif want bp-onecard "$L" 1 '^depth-ds41.sh: BLOOMERY_GEN_PLACE=bp is plan \(b′\), which loads on both cards .* it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000\+3090$' &&
  want bp-onecard "$L" 0 '^(ROW|FAIL) |^\[stub\] no lease'; then
  pass bp-onecard
fi

L=$tmp/twocard-xid.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" STUB_BENCH_XID=pp4 -- lcpp:6 lcpppp:4
if [ "$RC" != 1 ]; then
  fail twocard-xid "rc $RC, want 1" "$L"
elif want twocard-xid "$L" 1 '^ROW r1 lcpp d=6 ' &&
  want twocard-xid "$L" 1 "^FAIL r1 lcpppp p=4 rc=0 \| two cards: 1 NVRM Xid line\(s\) since the last arm.s check \(A6000 0, 3090 1 since the lease was taken\); last: .*Xid \(PCI:0000:41:00\): 79, " &&
  want twocard-xid "$L" 1 '^failed arms: r1 lcpppp p=4 rc=0; $'; then
  pass twocard-xid
fi

L=$tmp/twocard-dry.log
stub_run "$L" BLOOMERY_DRY=1 "$TC" -- lcpp:6
if [ "$RC" != 0 ]; then
  fail twocard-dry "rc $RC, want 0" "$L"
elif want twocard-dry "$L" 1 '^\[dry\] model=.* card=A6000\+3090 .* CUDA_VISIBLE_DEVICES=GPU-8c129fa6-[^,]*,GPU-307fa0f6-[^ ]* place=a ' &&
  want twocard-dry "$L" 1 '^\[dry\] two-card precheck: ok$' &&
  want twocard-dry "$L" 1 "^\[dry\] lcpp:6: timeout --kill-after=10 60 env  [^ ]*/lcpp-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -ngl 999 --n-cpu-moe 1 -fa on -t 4 -ts 1.5/1.5 --progress "; then
  pass twocard-dry
fi
SRVROW='\| llama-server ids=lcg: warm-up 20.00 tok/s majflt 0, continuation same \| prompt_n 6 prompt tok/s 60.00 \| build 1 \(stub\) \| device Stub Card \| majflt 0 \(timed 0; ≤ 0.0 % of W 0.2500 s\) \| wall [0-9]+s$'
L=$tmp/srv.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 -- 6 lcppsrv:6 4 lcppsrvpp:4 lcppsrvpp8:4
lcg6=$(awk -v n=6 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}')
if [ "$RC" != 0 ]; then
  fail srv "rc $RC, want 0" "$L"
elif [ -n "$(srv_left)" ]; then
  fail srv "servers still up: $(srv_left)" "$L"
elif want srv "$L" 1 "^ROW r1 lcppsrv d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) $SRVROW" &&
  want srv "$L" 1 '^ROW r1 lcppsrvpp p=4 n=0 \| tok/s\(pp\) 40.00 @ n=0, prompt 4, A6000 \(stub\) \| llama-server ids=lcg: warm-up 40.00 tok/s\(pp\) majflt 0, continuation same \| ub 512 b 2048 \(llama-server defaults\) \| build 1 \(stub\) \| device Stub Card \| majflt 0 \(timed 0; ≤ 0.0 % of W 0.1000 s\) \| wall ' &&
  want srv "$L" 1 '^ROW r1 lcppsrvpp8 p=4 n=0 .*\| ub 8 b 2048 \(the arm.s lever\) \|' &&
  want srv "$L" 1 '^ratio d=6 +ours/lcppsrv ' &&
  want srv "$L" 1 '^ratio pp p=4 +ours/lcppsrvpp ' &&
  want srv "$L" 1 '^ratio pp p=4 +ours/lcppsrvpp8 ' &&
  want srv "$L" 1 '^\[config\] lcppsrv: .*/llama-server -ngl 999 --n-cpu-moe 1 -fa on -t 4 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c ' &&
  want srv "$L" 5 '^    lcppsrv: .*/llama-server sha256=' &&
  want "srv argv" "$tmp/tmp/stub-srv-argv" 2 "^-m [^ ]+ -ngl 999 --n-cpu-moe 1 -fa on -t 4 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 256 --host 127.0.0.1 --port 0$" &&
  want "srv argv" "$tmp/tmp/stub-srv-argv" 1 "^-m [^ ]+ -ngl 999 --n-cpu-moe 1 -fa on -t 4 -ub 8 -b 2048 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 256 " &&
  want "srv ids" "$tmp/tmp/stub-srv-reqs" 2 "^[12] 6 4 $lcg6$" &&
  want "srv ids" "$tmp/tmp/stub-srv-reqs" 4 "^[12] 4 1 100000,"; then
  pass srv
fi

L=$tmp/srv-blocks.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks -- 6 lcppsrv:6 lcppsrvpp:4
if [ "$RC" != 0 ]; then
  fail srv-blocks "rc $RC, want 0" "$L"
elif want srv-blocks "$L" 1 '^\[config\] block 2/2 lcppsrv: lcppsrv:6 lcppsrvpp:4; discard lcppsrv:6, the longest prompt of the block.s arms, 6 ids$' &&
  want srv-blocks "$L" 1 '^DISCARD r0 lcppsrv d=6 ' &&
  want srv-blocks "$L" 2 '^ROW r1 lcppsrv(pp)? '; then
  pass srv-blocks
fi

for c in "srv-exit STUB_SRV_EXIT=1 rc=5 \| llama-server exited 5 before it answered /health" \
  "srv-hang STUB_SRV_HANG=1 rc=124 \| llama-server did not answer /health within 4 s" \
  "srv-badn STUB_SRV_BADN=1 rc=0 \| timed: predicted_n 3, not 4"; do
  name=${c%% *} rest=${c#* } envv=${rest%% *} pat=${rest#* }
  L=$tmp/$name.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_ARM_BOUND=4 "$envv" -- lcppsrv:6
  if [ "$RC" != 1 ]; then
    fail "$name" "rc $RC, want 1" "$L"
  elif [ -n "$(srv_left)" ]; then
    fail "$name" "servers still up: $(srv_left)" "$L"
  elif want "$name" "$L" 1 "^FAIL r1 lcppsrv d=6 $pat.* \| full output: " &&
    want "$name" "$L" 0 '^ROW ' &&
    want "$name" "$L" 1 '^    dropped: lcppsrv at 6$'; then
    pass "$name"
  fi
done

L=$tmp/srv-nobin.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 LCPPSRV="$T/bin/no-server" -- lcppsrv:6
if [ "$RC" != 2 ]; then
  fail srv-nobin "rc $RC, want 2" "$L"
elif want srv-nobin "$L" 1 "^depth-ds41.sh: the lcppsrv arms: no llama-server at $T/bin/no-server "; then
  pass srv-nobin
fi
L=$tmp/srv-probe.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_SRV_HELP_MISSING=--cache-ram -- lcppsrv:6
if [ "$RC" != 64 ]; then
  fail srv-probe "rc $RC, want 64" "$L"
elif want srv-probe "$L" 1 "^depth-ds41.sh: arm 'lcppsrv:6': .*/llama-server --help lists no --cache-ram$"; then
  pass srv-probe
fi

L=$tmp/srv-corpus.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 -- code:4 lcppsrv:code:4 4
if [ "$RC" != 0 ]; then
  fail srv-corpus "rc $RC, want 0" "$L"
elif want srv-corpus "$L" 1 '^ROW r1 lcppsrv@code d=4 n=4 .*\| llama-server ids=code: ' &&
  want srv-corpus "$L" 1 '^ratio code d=4 +code/lcppsrv@code ' &&
  want srv-corpus "$L" 0 '^ratio d=4 +ours/lcppsrv' &&
  want "srv-corpus ids" "$tmp/tmp/stub-srv-reqs" 2 '^[12] 4 4 101,102,103,104$'; then
  pass srv-corpus
fi

L=$tmp/srv-fit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 -- lcppsrvfit:6
if [ "$RC" != 0 ]; then
  fail srv-fit "rc $RC, want 0" "$L"
elif want srv-fit "$L" 1 "^ROW r1 lcppsrvfit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) $FIT_ROW llama-server ids=lcg: " &&
  want srv-fit "$L" 1 '^    lcppsrvfit fit load_tensors: +CPU_Mapped model buffer size = +2.00 MiB$' &&
  want "srv-fit argv" "$tmp/tmp/stub-srv-argv" 1 '^-m [^ ]+ -fa on -t 4 -fit on -fitt 1024 -v -np 1 '; then
  pass srv-fit
fi

L=$tmp/srv-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- lcppsrv:6 lcppsrvpp:4
if [ "$RC" != 0 ]; then
  fail srv-dry "rc $RC, want 0" "$L"
elif want srv-dry "$L" 1 "^\[dry\] lcppsrv:6: timeout --kill-after=10 60 [^ ]*/llama-server -m [^ ]* -ngl 999 --n-cpu-moe 1 -fa on -t 4 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 256 --host 127.0.0.1 --port 0   # row label 'lcppsrv', ids=lcg \(6 ids\): one POST /completion discarded, then the same timed, n_predict 4, " &&
  want srv-dry "$L" 1 "^\[dry\] lcppsrvpp:4: .* n_predict 1, greedy, ignore_eos, cache_prompt off, ub 512 b 2048 \(llama-server defaults\)$" &&
  [ ! -f "$tmp/tmp/stub-srv-argv" ]; then
  pass srv-dry
elif [ -f "$tmp/tmp/stub-srv-argv" ]; then
  fail srv-dry "the dry run started a server" "$L"
fi

L=$tmp/srv-cold.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_WARM_ROWS=1 STUB_SRV_COLD=2 -- lcppsrv:6
if [ "$RC" != 0 ]; then
  fail srv-cold "rc $RC, want 0" "$L"
elif want srv-cold "$L" 1 '^COLD r1 lcppsrv d=6 n=4 .*\| majflt 1000000 \(timed 1000000; ≤ [0-9.]+ % of W 0.2500 s\) \| wall [0-9]+s \[cold\]$' &&
  want srv-cold "$L" 1 '^ROW r1 lcppsrv d=6 n=4 .* continuation same \(the retry\) \| .*\| majflt 1000000 \(timed 0; ≤ 0.0 % of W 0.2500 s\) \| wall [0-9]+s$' &&
  want srv-cold "$L" 1 '^mean lcppsrv d=6 .*\(n=1\)' &&
  want "srv-cold requests" "$tmp/tmp/stub-srv-reqs" 3 '^[123] 6 4 ' &&
  want srv-cold "$L" 1 '^warm rows: 1 COLD row\(s\) ran once more: 1 clean on the retry, 0 FAIL rc=cold$' &&
  want srv-cold "$L" 1 '^\[config\] warm rows: BLOOMERY_WARM_ROWS=1 '; then
  pass srv-cold
fi
L=$tmp/srv-cold2.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_WARM_ROWS=1 STUB_SRV_COLD=2,3 -- lcppsrv:6
if [ "$RC" != 1 ]; then
  fail srv-cold2 "rc $RC, want 1" "$L"
elif want srv-cold2 "$L" 1 '^COLD r1 lcppsrv d=6 ' &&
  want srv-cold2 "$L" 1 '^FAIL r1 lcppsrv d=6 rc=cold \| cold after warm-up and one retry \(timed 1000000; ≤ [0-9.]+ % of W 0.2500 s\)$' &&
  want srv-cold2 "$L" 0 '^ROW ' &&
  want srv-cold2 "$L" 1 '^    dropped: lcppsrv at 6$' &&
  want srv-cold2 "$L" 1 '^failed arms: r1 lcppsrv d=6 rc=cold; $'; then
  pass srv-cold2
fi

L=$tmp/warm.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_WARM_ROWS=1 STUB_COLD_GEN=4:2 STUB_COLD_BENCH=d6:1 -- 6 4 lcpp:6
want_seq="PRIME r1 ours d=6
ROW r1 ours d=6
PRIME r1 ours d=4
COLD r1 ours d=4
PRIME r1 ours d=4
ROW r1 ours d=4
COLD r1 lcpp d=6
ROW r1 lcpp d=6"
seq=$(grep -oE '^(DISCARD|WARMUP|PRIME|COLD|ROW|FAIL) r[0-9]+ [^ ]+ [dp]=[0-9]+' "$L")
if [ "$RC" != 0 ]; then
  fail warm "rc $RC, want 0" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail warm "the rows' heads: $(echo "$seq" | paste -sd'|' -)" "$L"
elif [ "$(paste -sd'|' - < "$LOADS")" != "6 6 4 4|4 4" ]; then
  fail warm "the processes' --arm lists: $(paste -sd'|' - < "$LOADS"), want 6 6 4 4|4 4" "$L"
elif want warm "$L" 1 '^COLD r1 ours d=4 .* \[cold\]$' &&
  want warm "$L" 1 '^COLD r1 lcpp d=6 .*\(timed 100000; .* \[cold\]$' &&
  want warm "$L" 1 '^mean ours d=4 .*\(n=1\)' &&
  want warm "$L" 1 '^mean lcpp d=6 .*\(n=1\)' &&
  want warm "$L" 1 '^cold rows: 0 of 3 ' &&
  want warm "$L" 1 '^warm rows: 2 COLD row\(s\) ran once more: 2 clean on the retry, 0 FAIL rc=cold$'; then
  pass warm
fi
L=$tmp/warm-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_WARM_ROWS=1 STUB_COLD_GEN=4:4 STUB_COLD_BENCH=d6:2 -- 6 4 lcpp:6
if [ "$RC" != 1 ]; then
  fail warm-fail "rc $RC, want 1" "$L"
elif want warm-fail "$L" 1 '^FAIL r1 ours d=4 rc=cold \| cold after warm-up and one retry \(timed 100000; ' &&
  want warm-fail "$L" 1 '^FAIL r1 lcpp d=6 rc=cold \| cold after warm-up and one retry \(timed 100000; ' &&
  want warm-fail "$L" 1 '^    dropped: ours at 4$' &&
  want warm-fail "$L" 1 '^    dropped: lcpp at 6$' &&
  want warm-fail "$L" 1 '^ROW r1 ours d=6 ' &&
  want warm-fail "$L" 1 '^failed arms: r1 ours d=4 rc=cold; r1 lcpp d=6 rc=cold; $'; then
  pass warm-fail
fi
L=$tmp/warm-bad.log
stub_run "$L" BLOOMERY_WARM_ROWS=yes -- 6
if [ "$RC" != 64 ]; then
  fail warm-bad "rc $RC, want 64" "$L"
elif want warm-bad "$L" 1 "^depth-ds41.sh: BLOOMERY_WARM_ROWS is 0 .* or 1 .*, got 'yes'$"; then
  pass warm-bad
fi

if [ "${DEPTH_DS41_STUB_SHOW:-}" = 1 ]; then
  for L in "$tmp"/failed-arm.log "$tmp"/corpus.log "$tmp"/place.log "$tmp"/blocks.log "$tmp"/order-bad.log \
    "$tmp"/blocks-dry.log "$tmp"/blocks-ph.log "$tmp"/grouped.log "$tmp"/group-fail.log "$tmp"/load-arm.log \
    "$tmp"/solo.log "$tmp"/grouped-dry.log "$tmp"/fit.log "$tmp"/fit-preheat.log "$tmp"/fit-nobench.log \
    "$tmp"/fit-fail.log "$tmp"/fit-dry.log "$tmp"/twocard*.log "$tmp"/bp-onecard.log "$tmp"/srv*.log "$tmp"/warm*.log; do
    echo "--- ${L##*/} (rc of the run: see its last lines)"
    cat "$L"
  done
fi
echo "depth-ds41-stub: $n checks, $failed failed"
[ "$failed" = 0 ]
