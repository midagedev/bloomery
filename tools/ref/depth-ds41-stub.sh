#!/usr/bin/env bash
# The depth-ds41.sh stub test: the runner's arm loop with no lease, no card and no model. It copies the
# runner (DEPTH_DS41_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# timing-card.sh, cards.sh, lease-probe.sh, tdist.py, gguf-ranges.py and tools/bloomery, and a copy of
# lease.sh whose lease_take is replaced by a line that takes nothing; ref-paths.sh there is a stub
# profile whose engines are stub scripts (llama-bench, generate_ds41, nvidia-smi) and whose MODEL is
# gguf-ranges.py's two-shard fixture. Nothing it starts loads a model or touches a card.
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
  "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" "$ROOT/tools/ref/gguf-ranges.py" "$T/tools/ref/"
cp -R "$ROOT/tools/bloomery/records.py" "$ROOT/tools/bloomery/schema" "$T/tools/bloomery/"
echo 'lease_take() { echo "[stub] no lease: the stub test'"'"'s copy of lease.sh takes nothing"; }' >> "$T/tools/ref/lease.sh"
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
EOF
cat > "$T/bin/nvidia-smi" << 'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=name\ *) echo "NVIDIA RTX A6000 (stub)" ;;
  *--query-compute-apps*) ;;
  *) echo "stub, stub, stub, stub, stub, stub, stub, stub" ;;
esac
EOF
# The stub llama-bench: refuses --n-cpu-moe STUB_BENCH_FAIL_K; a pp run (-o json) prints one test with
# two samples; a decode run prints the markdown row under the engine's label, its model column naming
# the --n-cpu-moe it ran at. Under --progress it prints mainline's progress lines on stderr (the ik
# copy refuses the flag, as ik's llama-bench does), the error of a refused K last.
cat > "$T/bin/bench" << 'EOF'
#!/usr/bin/env bash
eng=${0##*/} k='' p='' n='' d='' gp='' json='' prog=''
while [ $# -gt 0 ]; do
  case $1 in
    --n-cpu-moe) k=$2; shift ;; -p) p=$2; shift ;; -n) n=$2; shift ;; -d) d=$2; shift ;; -gp) gp=$2; shift ;;
    -o) json=1; shift ;; --progress) prog=1 ;;
  esac
  shift
done
if [ -n "$prog" ] && [ "$eng" = ik-bench ]; then
  echo "error: unknown argument: --progress" >&2
  exit 1
fi
[ -z "$prog" ] || echo "llama-bench: benchmark 1/1: starting" >&2
if [ "$k" = "${STUB_BENCH_FAIL_K:-none}" ]; then
  echo "llama_init_from_model: failed to create context (stub: --n-cpu-moe $k)" >&2
  exit 1
fi
if [ -n "$json" ]; then
  if [ -n "$prog" ]; then
    for l in "warmup prompt run" "prompt run 1/2" "prompt run 2/2"; do echo "llama-bench: benchmark 1/1: $l" >&2; done
  fi
  echo "[{\"n_prompt\": $p, \"samples_ns\": [2000000000, 1000000000], \"build_commit\": \"stub\", \"build_number\": 0, \"gpu_info\": \"stub card\"}]"
  exit 0
fi
if [ -n "$prog" ]; then
  echo "llama-bench: benchmark 1/1: warmup generation run" >&2
  [ -z "$d" ] || echo "llama-bench: benchmark 1/1: depth run 1/1" >&2
  echo "llama-bench: benchmark 1/1: generation run 1/1" >&2
fi
if [ "$eng" = ik-bench ]; then label="tg${gp#*,}@pp${gp%,*}"; else label="tg$n @ d$d"; fi
echo "| model | size | test | t/s |"
echo "| --- | ---: | ---: | ---: |"
echo "| stub k=$k | 1 | $label | 20.00 ± 0.01 |"
echo "build: stub (0)"
EOF
cp "$T/bin/bench" "$T/bin/ik-bench"
mv "$T/bin/bench" "$T/bin/lcpp-bench"
touch "$T/Cargo.toml"
# The stub generate_ds41: refuses depth STUB_GEN_FAIL_DEPTH the first time (a marker file), and names
# STUB_GEN_PLACE in its SMOKE footer when set.
cat > "$T/target/release/generate_ds41" << 'EOF'
#!/usr/bin/env bash
depth='' n=32 place=a tokens=''
while [ $# -gt 0 ]; do
  case $1 in
    --depth) depth=$2; shift ;; --tokens) tokens=$2; shift ;; -n) n=$2; shift ;; --place) place=$2; shift ;;
    --warm) shift ;;
  esac
  shift
done
[ -z "$tokens" ] || depth=$(echo "$tokens" | tr ',' '\n' | grep -c .)
mark=${TMPDIR:-/tmp}/stub-gen-failed-$depth
if [ "$depth" = "${STUB_GEN_FAIL_DEPTH:-none}" ] && [ ! -e "$mark" ]; then
  touch "$mark"
  echo "plan place=$place (stub)"
  echo "error: the stub refuses depth $depth once" >&2
  exit 3
fi
echo "fed ids=$depth first=[1,2,3,4] last=[5,6,7,8] depth_sequence_from=0"
echo "time prompt n=$depth ms=100.0000 tok/s=$((depth * 10)).00 passes=1 kind=batch"
for i in $(seq 0 $((n - 1))); do echo "step $i $((depth + i)) $((1000 + i))"; done
for i in $(seq 1 $((n - 1))); do echo "time step $i ms=33.0000"; done
echo "SMOKE mode=graph place=${STUB_GEN_PLACE:-$place} prompt_tokens=0 depth=$depth generated=$n warm=0 steps=$((n - 1)) p50_ms=33.0000 mean_ms=33.0000 tok/s(p50)=30.30"
EOF
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
  rm -f "$tmp/tmp"/stub-gen-failed-*
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 \
    BLOOMERY_CPU_BUSY_COMMS=none "${e[@]}" bash tools/ref/depth-ds41.sh "$@") > "$log" 2>&1
  RC=$?
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
  want failed-arm "$L" 1 '^failed arms: r1 lcpp2 d=6 rc=1; r1 ours d=4 rc=3; r2 lcpp2 d=6 rc=1; $'; then
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

if [ "${DEPTH_DS41_STUB_SHOW:-}" = 1 ]; then
  for L in "$tmp"/failed-arm.log "$tmp"/corpus.log "$tmp"/place.log "$tmp"/blocks.log "$tmp"/order-bad.log \
    "$tmp"/blocks-dry.log "$tmp"/blocks-ph.log; do
    echo "--- ${L##*/} (rc of the run: see its last lines)"
    cat "$L"
  done
fi
echo "depth-ds41-stub: $n checks, $failed failed"
[ "$failed" = 0 ]
