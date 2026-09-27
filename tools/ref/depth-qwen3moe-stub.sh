#!/usr/bin/env bash
# The depth-qwen3moe.sh stub test: the runner's arm loop with no lease, no card and no model. It copies the
# runner (DEPTH_QWEN3MOE_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# timing-card.sh, cards.sh, lease-probe.sh, tdist.py, load-groups.sh, lcpp-fit.sh and cold-blocks.sh, and
# a copy of lease.sh whose lease_take is replaced by a line that takes nothing; ref-paths.sh there is a
# stub qwen4exp profile whose engines are stub scripts (llama-bench as lcpp and ik, generate_qwen3moe,
# mistralrs, nvidia-smi). Nothing it starts loads a model or touches a card. depth-ds41-stub.sh is the
# same for depth-ds41.sh; the two runners share cold-blocks.sh and load-groups.sh, not their engines.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/depth-qwen3moe-stub.sh'
#   ... 'DEPTH_QWEN3MOE_RUNNER=/tmp/base-depth-qwen3moe.sh bash tools/ref/depth-qwen3moe-stub.sh'   # FAIL-first
#
# Runs on the box (bash 4 or later, GNU timeout). One line per check, `ok <name>` or `FAIL <name>: <why>`
# followed by the run's output; exit 0 iff none failed. DEPTH_QWEN3MOE_STUB_SHOW=1 prints every run's
# whole output after the checks. The first three pin what the runner did before the cold tag and the
# blocks (green on the runner before them); the rest are the new paths (red on it).
#   rotate       6 lcpp:6 lcpppp:4 ik:6 mrs:6 mrspp:4 bin:<stub base>:6, two rounds, the defaults: the
#                rows' heads in the rotated order, no WARMUP or DISCARD row, rc 0.
#   fields       the same run: every row's fields up to `| wall <s>s` as they were, with nothing or
#                ` | …` after them.
#   rotate-dry   6 lcpp:6 512 lcpppp:512 under BLOOMERY_DRY=1: our arm's command line and each round's
#                order and loads as they were, and no warm-up line.
#   majflt       the rotate run: every row ends in its majflt column — ours timed from its prompt_ids
#                line, lcpp, lcpppp, mrs and mrspp timed from their progress lines, ik and the bin: arm
#                the whole process — and the closing summary counts the cold rows.
#   mrs-noiter   a mistralrs that prints no `Iteration 1/1...`: the row names it (`timed ? (no progress
#                line: …)`), never a count from nowhere.
#   progress-dry the same dry run: the lcpp and lcpppp lines carry --progress and their measured window.
#   warmup-rotate  BLOOMERY_AB_WARMUP=1 under rotate, lcpp:6 6, one round: a WARMUP r0 row of lcpp:6
#                in no mean.
#   blocks       BLOOMERY_AB_ORDER=blocks, 6 lcpp:6 4 lcpppp:4 ik:6 ikdef:6 lcppfit:6 lcppppfit8:4 mrs:6
#                mrspp:4, two rounds: five blocks (ours, lcpp, ik, lcppfit, mrs) in the order given, each
#                opened by one DISCARD r0 row — ours its longest prompt, lcpp its most draws (lcpp:6,
#                13), ik ik:6 (14 draws, the tie's first) at the block's largest --n-cpu-moe (ikdef's 2,
#                which the stub echoes in its table), the fit block lcppfit:6, mrs mrs:6 (20 ids) — and
#                each block's rows rotated by one slot a round; the rows' heads are compared whole. The
#                discards are in no mean.
#   order-bad    BLOOMERY_AB_ORDER=sideways is refused by name (rc 64).
#   warmup-bad   BLOOMERY_AB_WARMUP=2 is refused by name (rc 64).
#   blocks-dry   the blocks run's arms under BLOOMERY_DRY=1: the block plan, the ik discard's command line
#                at --n-cpu-moe 2, each block's rotation, and no plain round lines.
# The failures, red on the runner before FAIL rows (it stopped at the first failed arm, rc 1):
#   ref-fail     6 lcpp:6 lcpppp:4 lcppfit:6 ik:6 mrs:6 mrspp:4, two rounds; in round 1 lcpppp:4 aborts (rc
#                134), lcppfit:6's fit never runs and mrs:6 prints no row: three FAIL rows naming why, the
#                last line and the full output's file (which holds the abort), every other arm's row after
#                them, the three labels dropped from the means and ratios by name, `failed arms: …`, rc 1.
#   discard-fail BLOOMERY_AB_ORDER=blocks, lcpp:6 lcppfit:6 lcppppfit8:8 mrs:6, one round: each block's
#                discard fails (lcpp:6 no row, lcppppfit8:8 aborts — the 2026-09-28 sitting's shape — mrs:6
#                no row) as `FAIL r0 …`, and every block's round still runs; nothing drops; rc 1.
#   discard-nofit  the fit block's discard lcppfit:6 with no fit: `FAIL r0 lcppfit`, the block's rows after it.
#   warmup-fail  BLOOMERY_AB_WARMUP=1 under rotate, the warm-up lcpp:6 aborts: `FAIL r0 lcpp`, the rows after.
#   group-fail   6 4 5 in one load, depth 4 ending the process once: 6's row, 4's FAIL row, the driver's
#                `[load]` line, and 5 in a fresh load (the processes' --arm lists 6 4 5, then 5); rc 1.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNNER=${DEPTH_QWEN3MOE_RUNNER:-$ROOT/tools/ref/depth-qwen3moe.sh}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/depth-qwen3moe-stub.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
T=$tmp/tree
mkdir -p "$T/tools/ref" "$T/target/release" "$T/base/target/release" "$T/bin" "$tmp/tmp"
cp "$RUNNER" "$T/tools/ref/depth-qwen3moe.sh"
cp "$ROOT/tools/ref/timing-card.sh" "$ROOT/tools/ref/cards.sh" "$ROOT/tools/ref/lease-probe.sh" \
  "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" "$ROOT/tools/ref/load-groups.sh" \
  "$ROOT/tools/ref/lcpp-fit.sh" "$ROOT/tools/ref/cold-blocks.sh" "$T/tools/ref/"
echo 'lease_take() { echo "[stub] no lease: the stub test'"'"'s copy of lease.sh takes nothing"; }' >> "$T/tools/ref/lease.sh"
touch "$T/model-00001-of-00001.gguf"
cat > "$T/tools/ref/ref-paths.sh" << EOF
# shellcheck shell=bash
MODEL_NAME=qwen4exp
MODEL=$T/model-00001-of-00001.gguf
IK=$T IKBIN=$T/bin/ik-bench LCPP=$T LCPPBIN=$T/bin/lcpp-bench MRS=$T MRSBIN=$T/bin/mistralrs
IK_GPU_FLAGS='-ngl 99 --n-cpu-moe 1' IK_GPU_DEFAULT_FLAGS='-ngl 99 --n-cpu-moe 2'
LCPP_GPU_FLAGS='-ngl 99 -fa on -ncmoe 1'
MRS_FLAGS='--format gguf'
EOF
cat > "$T/bin/nvidia-smi" << 'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=name\ *) echo "NVIDIA RTX A6000 (stub)" ;;
  *--query-compute-apps*) ;;
  *) echo "stub, stub, stub, stub, stub, stub, stub, stub" ;;
esac
EOF
# The stub llama-bench at -r 1: a markdown row under the engine's label (pp<P> at 400.00 t/s, a decode
# test at 20.00), its model column naming the --n-cpu-moe (or -ncmoe) it ran at. Under --progress it
# prints mainline's progress lines on stderr (the ik copy refuses the flag, as ik's llama-bench does).
# --help lists --fit-target; under -fitt -v it prints two model loads on stderr, the fit's measuring
# one and the real one with two expert tensors of blk 1 overridden to the host. The failures, each the
# first time only a test of that label runs (a marker file per failure and label): the label
# STUB_BENCH_ABORT names aborts after its progress lines as ggml's CUDA error does (rc 134, no table
# row), STUB_BENCH_NOVAL's prints its table without its row (rc 0), and under -fitt STUB_BENCH_NOFIT's
# prints no model load (the fit never ran).
cat > "$T/bin/bench" << 'EOF'
#!/usr/bin/env bash
eng=${0##*/} k='' p=0 n=0 d='' gp='' prog='' fitt='' verb=''
while [ $# -gt 0 ]; do
  case $1 in
    --n-cpu-moe | -ncmoe) k=$2; shift ;; -p) p=$2; shift ;; -n) n=$2; shift ;; -d) d=$2; shift ;; -gp) gp=$2; shift ;;
    --progress) prog=1 ;; -fitt) fitt=$2; shift ;; -v) verb=1 ;;
    -h | --help)
      echo "usage: $eng [options]"
      echo "  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)"
      exit 0
      ;;
  esac
  shift
done
if [ "$p" != 0 ]; then
  label="pp$p" val=400.00
elif [ "$eng" = ik-bench ]; then
  label="tg${gp#*,}@pp${gp%,*}" val=20.00
else
  label="tg$n @ d$d" val=20.00
fi
# once <failure>: true the first time a test of this label meets <failure> (named by STUB_BENCH_<failure>).
once() {
  local var=STUB_BENCH_$1 m
  [ "${!var:-}" = "$label" ] || return 1
  m=${TMPDIR:-/tmp}/stub-once-bench-$1-${label//[^A-Za-z0-9]/_}
  [ ! -e "$m" ] || return 1
  touch "$m"
}
if [ -n "$fitt" ] && [ -n "$verb" ] && ! once NOFIT; then
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
if [ -n "$prog" ] && [ "$eng" = ik-bench ]; then
  echo "error: unknown argument: --progress" >&2
  exit 1
fi
if [ -n "$prog" ]; then
  echo "llama-bench: benchmark 1/1: starting" >&2
  if [ "$p" != 0 ]; then
    for l in "warmup prompt run" "prompt run 1/1"; do echo "llama-bench: benchmark 1/1: $l" >&2; done
  else
    echo "llama-bench: benchmark 1/1: warmup generation run" >&2
    [ -z "$d" ] || echo "llama-bench: benchmark 1/1: depth run 1/1" >&2
    echo "llama-bench: benchmark 1/1: generation run 1/1" >&2
  fi
fi
if once ABORT; then
  echo "CUDA error: an illegal memory access was encountered (stub)" >&2
  echo "ggml_cuda_error: in function ggml_backend_cuda_graph_compute (stub)" >&2
  exit 134
fi
echo "| model | size | test | t/s |"
echo "| --- | ---: | ---: | ---: |"
once NOVAL || echo "| stub k=$k | 1 | $label | $val ± 0.01 |"
echo "build: stub (0)"
EOF
cp "$T/bin/bench" "$T/bin/ik-bench"
mv "$T/bin/bench" "$T/bin/lcpp-bench"
# The stub mistralrs: --version, and `bench` printing its timed iteration's log line (not under
# STUB_MRS_NO_ITER) and a box-drawn row: TTFT for --gen-len 1 (400.0 T/s), else the decode row (50.0).
# Under STUB_MRS_NOVAL its first bench prints no row (a marker file) and exits 0.
cat > "$T/bin/mistralrs" << 'EOF'
#!/usr/bin/env bash
[ "${1:-}" != --version ] || { echo "mistralrs 0.0.0-stub"; exit 0; }
p=0 g=0 d=0
while [ $# -gt 0 ]; do
  case $1 in --prompt-len) p=$2; shift ;; --gen-len) g=$2; shift ;; --depth) d=$2; shift ;; esac
  shift
done
echo "2026-01-01T00:00:00Z  INFO mistralrs_cli: Layers 0-1: cuda[0]"
echo "2026-01-01T00:00:00Z  INFO mistralrs_cli: Warmup complete."
[ -n "${STUB_MRS_NO_ITER:-}" ] || echo "2026-01-01T00:00:00Z  INFO mistralrs_cli: Iteration 1/1..."
m=${TMPDIR:-/tmp}/stub-once-mrs-noval
if [ -n "${STUB_MRS_NOVAL:-}" ] && [ ! -e "$m" ]; then
  touch "$m"
  echo "2026-01-01T00:00:00Z  WARN mistralrs_cli: the stub prints no row once"
elif [ "$g" = 1 ]; then
  echo "│ TTFT ($p input tokens) ┆ 400.0 ± 0.0 ┆ 10.00 ms │"
else
  echo "│ Decode ($g tokens @ d$d) ┆ 50.0 ± 0.1 ┆ 20.00 ms TPOT │"
fi
EOF
touch "$T/Cargo.toml"
# The stub generate_qwen3moe: a one-arm run (--tokens) prints its prompt ids before its load lines; an
# --arm list prints each arm's `arm` line, waits for a line on stdin under --arm-sync, then its prompt ids.
# Each arm: the step-0, time prompt (P x 10 tok/s, kind=gemm) and step lines, and a SMOKE footer at 5 ms
# a step. Every process appends its arm list to $TMPDIR/stub-gen-loads. An arm of depth
# STUB_GEN_FAIL_DEPTH ends the process at rc 3 the first time (a marker file), after its prompt ids.
cat > "$T/target/release/generate_qwen3moe" << 'EOF'
#!/usr/bin/env bash
tokens='' n=32 ctx=0 sync='' arms=()
while [ $# -gt 0 ]; do
  case $1 in
    --tokens) tokens=$2; shift ;; -n) n=$2; shift ;; --ctx) ctx=$2; shift ;; --warm) shift ;;
    --arm) arms+=("$2"); shift ;; --arm-sync) sync=1 ;;
  esac
  shift
done
count() { echo "$1" | tr ',' '\n' | grep -c .; }
listed=1
if [ ${#arms[@]} -eq 0 ]; then listed='' arms=("$tokens"); echo "prompt_ids [$tokens]"; fi
echo "$(for a in "${arms[@]}"; do printf '%s ' "$(count "$a")"; done)" >> "${TMPDIR:-/tmp}/stub-gen-loads"
echo "load arch=stub ctx=$ctx (stub)"
echo "capture graph_nodes=10"
for k in "${!arms[@]}"; do
  depth=$(count "${arms[$k]}")
  if [ -n "$listed" ]; then
    echo "arm i=$k arms=${#arms[@]} ids=$depth n=$n"
    if [ -n "$sync" ]; then read -r _ || { echo "error: stdin closed before arm $k" >&2; exit 65; }; fi
    echo "prompt_ids [${arms[$k]}]"
  fi
  mark=${TMPDIR:-/tmp}/stub-once-gen-$depth
  if [ "$depth" = "${STUB_GEN_FAIL_DEPTH:-none}" ] && [ ! -e "$mark" ]; then
    touch "$mark"
    echo "error: the stub refuses depth $depth once" >&2
    exit 3
  fi
  echo "step 0 $depth 1000 (stub)"
  echo "time prompt n=$depth ms=100.0000 tok/s=$((depth * 10)).00 passes=1 kind=gemm"
  echo "stat prompt ubatch_tokens=0 (no ubatch ran)"
  for i in $(seq 1 $((n - 1))); do echo "step $i $((depth + i)) $((1000 + i))"; echo "time step $i ms=5.0000"; done
  echo "SMOKE mode=graph prompt_tokens=$depth depth=$depth seeded=false generated=$n warm=0 steps=$((n - 1)) p50_ms=5.0000 mean_ms=5.0000 tok/s(p50)=200.00 ctx=$ctx"
done
EOF
cp "$T/target/release/generate_qwen3moe" "$T/base/target/release/generate_qwen3moe"
chmod +x "$T/bin/"* "$T/target/release/generate_qwen3moe" "$T/base/target/release/generate_qwen3moe"
BASEBIN=$T/base/target/release/generate_qwen3moe

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
  rm -f "$tmp/tmp/stub-gen-loads" "$tmp/tmp"/stub-once-*
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 \
    "${e[@]}" bash tools/ref/depth-qwen3moe.sh "$@") > "$log" 2>&1
  RC=$?
}
# want <name> <log> <count> <pattern>: the log holds exactly <count> lines matching grep -E <pattern>.
want() {
  local c
  c=$(grep -cE -- "$4" "$2")
  [ "$c" = "$3" ] || { fail "$1" "$c lines match /$4/, want $3" "$2"; return 1; }
}
heads() { grep -oE '^(DISCARD|WARMUP|ROW|FAIL) r[0-9]+ [^ ]+ [dp]=[0-9]+' "$1"; }

L=$tmp/rotate.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 -- 6 lcpp:6 lcpppp:4 ik:6 mrs:6 mrspp:4 "bin:$BASEBIN:6"
want_seq="ROW r1 ours d=6
ROW r1 lcpp d=6
ROW r1 lcpppp p=4
ROW r1 ik d=6
ROW r1 mrs d=6
ROW r1 mrspp p=4
ROW r1 bin:base d=6
ROW r2 lcpp d=6
ROW r2 lcpppp p=4
ROW r2 ik d=6
ROW r2 mrs d=6
ROW r2 mrspp p=4
ROW r2 bin:base d=6
ROW r2 ours d=6"
seq=$(heads "$L")
if [ "$RC" != 0 ]; then
  fail rotate "rc $RC, want 0" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail rotate "the rows' heads are not the rotation: $(echo "$seq" | paste -sd'|' -)" "$L"
elif want rotate "$L" 0 '^(WARMUP|DISCARD) '; then
  pass rotate
fi

END='\| wall [0-9]+s( \| .*)?$'
if [ "$RC" != 0 ]; then
  fail fields "the rotate run failed (rc $RC)" "$L"
elif want fields "$L" 2 "^ROW r[12] ours d=6 n=4 ctx=256 \| tok/s\(mean\) 200.00 @ n=4, depth 6, A6000 \(stub\) \| p50 5.0000 ms \| mean 5.0000 ms \| tok/s\(p50\) 200.00 \| warm 0 \| first10_p50 5.0000 \| last10_p50 5.0000 \| distinct_tokens 3 \| nodes 10 \| pp_tok/s 60.00 \(n=6, passes=1, kind=gemm\) \| slot 1/1 $END" &&
  want fields "$L" 2 "^ROW r[12] bin:base d=6 n=4 ctx=256 \| tok/s\(mean\) 200.00 @ n=4, depth 6, A6000 \(stub\) \| p50 5.0000 ms .* \| nodes 10 \| pp_tok/s 60.00 \(n=6, passes=1, kind=gemm\) $END" &&
  want fields "$L" 2 "^ROW r[12] lcpp d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) \| build stub \(0\) \| device \? $END" &&
  want fields "$L" 2 "^ROW r[12] lcpppp p=4 n=0 \| tok/s\(pp\) 400.00 @ n=0, prompt 4, A6000 \(stub\) \| ub 512 b 2048 \(llama-bench defaults\) \| build stub \(0\) \| device \? $END" &&
  want fields "$L" 2 "^ROW r[12] ik d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) \| build stub \(0\) \| device \? $END" &&
  want fields "$L" 2 "^ROW r[12] mrs d=6 n=4 \| tok/s 50.0 @ n=4, depth 6, A6000 \(stub\) \| build mistralrs 0.0.0-stub \| device layers 0-1: cuda\[0\] $END" &&
  want fields "$L" 2 "^ROW r[12] mrspp p=4 n=0 \| tok/s\(pp\) 400.0 @ n=0, prompt 4, A6000 \(stub\) \| scheduler defaults: [^|]* \| build mistralrs 0.0.0-stub \| device layers 0-1: cuda\[0\] $END" &&
  want fields "$L" 1 '^ratio d=6 +ours/lcpp +mean 1[0-9.]+ ' &&
  want fields "$L" 1 '^other-busy rows: 0 of 14 '; then
  pass fields
fi

L=$tmp/rotate-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- 6 lcpp:6 512 lcpppp:512
if [ "$RC" != 0 ]; then
  fail rotate-dry "rc $RC, want 0" "$L"
elif want rotate-dry "$L" 1 '^\[dry\] 6: one arm of a load: timeout --kill-after=10 \$\(\(BOUND x arms \+ BOUND\)\) target/release/generate_qwen3moe --arm <lcg_prompt 6> \.\.\. -n 4 --ctx 256 --time --arm-sync   # load key target/release/generate_qwen3moe\|ctx=256$' &&
  want rotate-dry "$L" 1 '^\[dry\] round 1 order: 6 lcpp:6 512 lcpppp:512$' &&
  want rotate-dry "$L" 1 '^\[dry\] round 2 order: lcpp:6 512 lcpppp:512 6$' &&
  want rotate-dry "$L" 1 '^\[dry\] round 2 loads: lcpp:6 \[512\] lcpppp:512 \[6\]$' &&
  want rotate-dry "$L" 0 '^\[dry\] (warmup|block)'; then
  pass rotate-dry
fi

L=$tmp/rotate.log
MAJ='\| majflt [0-9]+ \('
if want majflt "$L" 14 "$END" &&
  want majflt "$L" 14 "^ROW .*\| wall [0-9]+s $MAJ.*; ≤ [0-9.]+ % of W [0-9.]+ s\)( \[cold\])?$" &&
  want majflt "$L" 2 "^ROW r[12] ours d=6 .*${MAJ}timed [0-9]+; ≤ [0-9.]+ % of W 0.1200 s\)" &&
  want majflt "$L" 4 "^ROW r[12] lcpp(pp)? [dp]=[46] .*${MAJ}timed [0-9]+; " &&
  want majflt "$L" 4 "^ROW r[12] mrs(pp)? [dp]=[46] .*${MAJ}timed [0-9]+; " &&
  want majflt "$L" 2 "^ROW r[12] ik d=6 .*${MAJ}whole process; ≤ [0-9.]+ % of W 0.2000 s\)" &&
  want majflt "$L" 2 "^ROW r[12] bin:base d=6 .*${MAJ}timed \? \(a one-arm run prints its prompt ids before its load: the whole process\); " &&
  want majflt "$L" 2 "^ROW r[12] mrs d=6 .*of W 0.0600 s\)" &&
  want majflt "$L" 1 '^cold rows: [0-9]+ of 14 ' &&
  want majflt "$L" 1 '^\[config\] cold tag: ' &&
  want majflt "$L" 1 '^mean ours d=6 .*\(n=2\)  \[cold [0-9]/2\]$'; then
  pass majflt
fi

L=$tmp/mrs-noiter.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_MRS_NO_ITER=1 -- mrs:6
if [ "$RC" != 0 ]; then
  fail mrs-noiter "rc $RC, want 0" "$L"
elif want mrs-noiter "$L" 1 "^ROW r1 mrs d=6 .*${MAJ}timed \? \(no progress line: the whole process\); "; then
  pass mrs-noiter
fi

L=$tmp/rotate-dry.log
if want progress-dry "$L" 1 "^\[dry\] lcpp:6: timeout --kill-after=10 60 [^ ]*/lcpp-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -ngl 99 -fa on -ncmoe 1 --progress   # row label 'tg4 @ d6', measured window from /: generation run 1/1\\\$/$" &&
  want progress-dry "$L" 1 "^\[dry\] lcpppp:512: timeout --kill-after=10 60 [^ ]*/lcpp-bench -m [^ ]* -p 512 -n 0 -r 1 -ngl 99 -fa on -ncmoe 1 --progress   # row label 'pp512', ub 512 b 2048 \(llama-bench defaults\), measured window from /: prompt run 1/1\\\$/$"; then
  pass progress-dry
fi

L=$tmp/warmup-rotate.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=1 -- lcpp:6 6
if [ "$RC" != 0 ]; then
  fail warmup-rotate "rc $RC, want 0" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "WARMUP r0 lcpp d=6|ROW r1 lcpp d=6|ROW r1 ours d=6" ]; then
  fail warmup-rotate "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want warmup-rotate "$L" 1 '^\[warmup\] lcpp:6 ran once before round 1' &&
  want warmup-rotate "$L" 1 '^mean lcpp d=6 .*\(n=1\)' &&
  want warmup-rotate "$L" 1 '^cold rows: [0-9]+ of 2 '; then
  pass warmup-rotate
fi

BLOCK_ARMS=(6 lcpp:6 4 lcpppp:4 ik:6 ikdef:6 lcppfit:6 lcppppfit8:4 mrs:6 mrspp:4)
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
ROW r2 lcpppp p=4
ROW r2 lcpp d=6
DISCARD r0 ik d=6
ROW r1 ik d=6
ROW r1 ikdef d=6
ROW r2 ikdef d=6
ROW r2 ik d=6
DISCARD r0 lcppfit d=6
ROW r1 lcppfit d=6
ROW r1 lcppppfit8 p=4
ROW r2 lcppppfit8 p=4
ROW r2 lcppfit d=6
DISCARD r0 mrs d=6
ROW r1 mrs d=6
ROW r1 mrspp p=4
ROW r2 mrspp p=4
ROW r2 mrs d=6"
seq=$(heads "$L")
if [ "$RC" != 0 ]; then
  fail blocks "rc $RC, want 0" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail blocks "the rows' heads are not the block order: $(echo "$seq" | paste -sd'|' -)" "$L"
elif want blocks "$L" 1 '^\[config\] order: blocks, a discard before each block: on$' &&
  want blocks "$L" 1 '^\[config\] block 1/5 ours: 6 4; discard 6, the longest prompt of the block.s arms, 6 ids$' &&
  want blocks "$L" 1 '^\[config\] block 2/5 lcpp: lcpp:6 lcpppp:4; discard lcpp:6, the most token draws of the block.s arms \(lcpp:6 13, lcpppp:4 8\)$' &&
  want blocks "$L" 1 '^\[config\] block 3/5 ik: ik:6 ikdef:6; discard ik:6 at --n-cpu-moe 2, the most token draws of the block.s arms \(ik:6 14, ikdef:6 14\), at the block.s largest --n-cpu-moe$' &&
  want blocks "$L" 1 '^\[config\] block 4/5 lcppfit: lcppfit:6 lcppppfit8:4; discard lcppfit:6, the most token draws of the block.s arms \(lcppfit:6 13, lcppppfit8:4 8\)$' &&
  want blocks "$L" 1 '^\[config\] block 5/5 mrs: mrs:6 mrspp:4; discard mrs:6, the most token draws of the block.s arms \(mrs:6 20, mrspp:4 10\)$' &&
  want blocks "$L" 5 '^\[block\] [1-5]/5 ' &&
  want blocks "$L" 5 '^\[discard\] ' &&
  want blocks "$L" 3 '^    ik(def)? table \| stub k=2 ' &&
  want blocks "$L" 2 '^    ik table \| stub k=1 ' &&
  want blocks "$L" 3 '^(DISCARD|ROW) r[0-2] lcppfit d=6 .*\| fit offloaded 3/3, overridden CPU:2 in blk 1-1 \(blk 1: 2\), ' &&
  want blocks "$L" 25 "$MAJ" &&
  want blocks "$L" 1 '^mean ours d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^mean lcpp d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^mean ik d=6 .*\(n=2\)' &&
  want blocks "$L" 1 '^cold rows: [0-9]+ of 20 ' &&
  want blocks "$L" 0 '^WARMUP '; then
  pass blocks
fi

L=$tmp/order-bad.log
stub_run "$L" BLOOMERY_AB_ORDER=sideways -- 6
if [ "$RC" != 64 ]; then
  fail order-bad "rc $RC, want 64" "$L"
elif want order-bad "$L" 1 "^depth-qwen3moe.sh: BLOOMERY_AB_ORDER is rotate .* or blocks .*, got 'sideways'$"; then
  pass order-bad
fi

L=$tmp/warmup-bad.log
stub_run "$L" BLOOMERY_AB_WARMUP=2 -- 6
if [ "$RC" != 64 ]; then
  fail warmup-bad "rc $RC, want 64" "$L"
elif want warmup-bad "$L" 1 "^depth-qwen3moe.sh: BLOOMERY_AB_WARMUP is 1 .* or 0 .*, got '2'$"; then
  pass warmup-bad
fi

L=$tmp/blocks-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_ORDER=blocks BLOOMERY_DRY=1 -- "${BLOCK_ARMS[@]}"
if [ "$RC" != 0 ]; then
  fail blocks-dry "rc $RC, want 0" "$L"
elif want blocks-dry "$L" 1 '^\[dry\] block 3/5 ik: ik:6 ikdef:6; discard ik:6 at --n-cpu-moe 2,' &&
  want blocks-dry "$L" 1 "^\[dry\] block 3 discard ik:6: timeout --kill-after=10 60 [^ ]*/ik-bench -m [^ ]* -p 0 -n 0 -gp 6,4 -r 1 -ngl 99 --n-cpu-moe 2   # row label 'tg4@pp6'$" &&
  want blocks-dry "$L" 1 '^\[dry\] block 2 round 2 order: lcpppp:4 lcpp:6$' &&
  want blocks-dry "$L" 1 '^\[dry\] block 1 round 1 loads: \[6 4\]$' &&
  want blocks-dry "$L" 1 '^\[dry\] block 1 round 2 order: 4 6$' &&
  want blocks-dry "$L" 1 '^\[dry\] block 5 discard mrs:6: ' &&
  want blocks-dry "$L" 1 '^\[dry\] warmup: the blocks. discards below' &&
  want blocks-dry "$L" 0 '^\[dry\] round '; then
  pass blocks-dry
fi

# The failures (red on the runner before FAIL rows: it exits 1 at the first one).
L=$tmp/ref-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 STUB_BENCH_ABORT=pp4 'STUB_BENCH_NOFIT=tg4 @ d6' STUB_MRS_NOVAL=1 -- \
  6 lcpp:6 lcpppp:4 lcppfit:6 ik:6 mrs:6 mrspp:4
want_seq="ROW r1 ours d=6
ROW r1 lcpp d=6
FAIL r1 lcpppp p=4
FAIL r1 lcppfit d=6
ROW r1 ik d=6
FAIL r1 mrs d=6
ROW r1 mrspp p=4
ROW r2 lcpp d=6
ROW r2 lcpppp p=4
ROW r2 lcppfit d=6
ROW r2 ik d=6
ROW r2 mrs d=6
ROW r2 mrspp p=4
ROW r2 ours d=6"
seq=$(heads "$L")
FULL="\| full output: [^ ]*/depth-qwen3moe-"
if [ "$RC" != 1 ]; then
  fail ref-fail "rc $RC, want 1" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail ref-fail "the rows' heads: $(echo "$seq" | paste -sd'|' -)" "$L"
elif want ref-fail "$L" 1 "^FAIL r1 lcpppp p=4 rc=134 \| no 'pp4' row; last line: ggml_cuda_error: in function ggml_backend_cuda_graph_compute \(stub\) ${FULL}lcpppp-p4-r1\.log$" &&
  want ref-fail "$L" 1 "^FAIL r1 lcppfit d=6 rc=0 \| llama-bench.s fit did not run: 0 model load\(s\) in its -v output, want the fit.s measuring load and the real one; last line: build: stub \(0\) ${FULL}lcppfit-d6-r1\.log$" &&
  want ref-fail "$L" 1 "^FAIL r1 mrs d=6 rc=0 \| no 'Decode \(4 tokens @ d6\)' row; last line: .*WARN mistralrs_cli: the stub prints no row once ${FULL}mrs-d6-r1\.log$" &&
  want ref-fail "$L" 11 '^ROW ' &&
  want ref-fail "$L" 1 '^=== dropped from the means and the ratios below' &&
  want ref-fail "$L" 3 '^    dropped: (lcpppp at 4|lcppfit at 6|mrs at 6)$' &&
  want ref-fail "$L" 0 '^mean (pp )?(lcpppp|lcppfit|mrs) ' &&
  want ref-fail "$L" 1 '^mean lcpp d=6 .*\(n=2\)' &&
  want ref-fail "$L" 1 '^mean pp mrspp p=4 .*\(n=2\)' &&
  want ref-fail "$L" 1 '^ratio d=6 +ours/ik ' &&
  want ref-fail "$L" 0 '^ratio d=6 +ours/(lcppfit|mrs) ' &&
  want ref-fail "$L" 1 '^cold rows: [0-9]+ of 11 ' &&
  want ref-fail "$L" 1 '^failed arms: 3 \(FAIL rows, ' &&
  want ref-fail "$L" 1 '^failed arms: r1 lcpppp p=4 rc=134; r1 lcppfit d=6 rc=0; r1 mrs d=6 rc=0; $'; then
  if grep -q '^ggml_cuda_error: ' "$tmp/tmp/depth-qwen3moe-lcpppp-p4-r1.log" 2> /dev/null; then
    pass ref-fail
  else
    fail ref-fail "the FAIL row's full output file does not hold the abort" "$L"
  fi
fi

L=$tmp/discard-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks STUB_BENCH_ABORT=pp8 'STUB_BENCH_NOVAL=tg4 @ d6' STUB_MRS_NOVAL=1 -- \
  lcpp:6 lcppfit:6 lcppppfit8:8 mrs:6
want_seq="FAIL r0 lcpp d=6
ROW r1 lcpp d=6
FAIL r0 lcppppfit8 p=8
ROW r1 lcppfit d=6
ROW r1 lcppppfit8 p=8
FAIL r0 mrs d=6
ROW r1 mrs d=6"
seq=$(heads "$L")
if [ "$RC" != 1 ]; then
  fail discard-fail "rc $RC, want 1" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail discard-fail "the rows' heads: $(echo "$seq" | paste -sd'|' -)" "$L"
elif want discard-fail "$L" 1 "^FAIL r0 lcpp d=6 rc=0 \| no 'tg4 @ d6' row; last line: build: stub \(0\) ${FULL}lcpp-d6-r0\.log$" &&
  want discard-fail "$L" 1 "^FAIL r0 lcppppfit8 p=8 rc=134 \| no 'pp8' row; last line: ggml_cuda_error: .* ${FULL}lcppppfit8-p8-r0\.log$" &&
  want discard-fail "$L" 1 "^FAIL r0 mrs d=6 rc=0 \| no 'Decode \(4 tokens @ d6\)' row; " &&
  want discard-fail "$L" 3 '^\[discard\] ' &&
  want discard-fail "$L" 0 '^=== dropped|^    dropped: ' &&
  want discard-fail "$L" 1 '^mean lcpp d=6 .*\(n=1\)' &&
  want discard-fail "$L" 1 '^mean pp lcppppfit8 p=8 .*\(n=1\)' &&
  want discard-fail "$L" 1 '^failed arms: 3 \(FAIL rows, ' &&
  want discard-fail "$L" 1 '^failed arms: r0 lcpp d=6 rc=0; r0 lcppppfit8 p=8 rc=134; r0 mrs d=6 rc=0; $'; then
  pass discard-fail
fi

L=$tmp/discard-nofit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_ORDER=blocks 'STUB_BENCH_NOFIT=tg4 @ d6' -- lcppfit:6 lcppppfit:4
if [ "$RC" != 1 ]; then
  fail discard-nofit "rc $RC, want 1" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "FAIL r0 lcppfit d=6|ROW r1 lcppfit d=6|ROW r1 lcppppfit p=4" ]; then
  fail discard-nofit "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want discard-nofit "$L" 1 "^FAIL r0 lcppfit d=6 rc=0 \| llama-bench.s fit did not run: 0 model load\(s\) " &&
  want discard-nofit "$L" 1 '^ROW r1 lcppfit d=6 .*\| fit offloaded 3/3, ' &&
  want discard-nofit "$L" 1 '^failed arms: r0 lcppfit d=6 rc=0; $'; then
  pass discard-nofit
fi

L=$tmp/warmup-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=1 'STUB_BENCH_ABORT=tg4 @ d6' -- lcpp:6 6
if [ "$RC" != 1 ]; then
  fail warmup-fail "rc $RC, want 1" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "FAIL r0 lcpp d=6|ROW r1 lcpp d=6|ROW r1 ours d=6" ]; then
  fail warmup-fail "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want warmup-fail "$L" 1 "^FAIL r0 lcpp d=6 rc=134 \| no 'tg4 @ d6' row; " &&
  want warmup-fail "$L" 1 '^\[warmup\] lcpp:6 ran once before round 1 and is discarded \(the WARMUP or FAIL r0 row above\)$' &&
  want warmup-fail "$L" 1 '^mean lcpp d=6 .*\(n=1\)' &&
  want warmup-fail "$L" 1 '^failed arms: r0 lcpp d=6 rc=134; $'; then
  pass warmup-fail
fi

L=$tmp/group-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_FAIL_DEPTH=4 -- 6 4 5
loads=$(sed 's/ *$//' "$tmp/tmp/stub-gen-loads" 2> /dev/null | paste -sd'|' -)
if [ "$RC" != 1 ]; then
  fail group-fail "rc $RC, want 1" "$L"
elif [ "$loads" != "6 4 5|5" ]; then
  fail group-fail "the processes' --arm lists: $loads, want 6 4 5|5" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "ROW r1 ours d=6|FAIL r1 ours d=4|ROW r1 ours d=5" ]; then
  fail group-fail "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want group-fail "$L" 1 '^ROW r1 ours d=6 .*\| slot 1/3 \|' &&
  want group-fail "$L" 1 "^FAIL r1 ours d=4 rc=3 \| exited 3; last line: error: the stub refuses depth 4 once ${FULL}ours-d4-r1\.log$" &&
  want group-fail "$L" 1 '^\[load\] r1: arm 4 failed \(rc 3\); the 1 arm\(s\) after it run in a fresh load$' &&
  want group-fail "$L" 1 '^ROW r1 ours d=5 .*\| slot 1/1 \|' &&
  want group-fail "$L" 1 '^    dropped: ours at 4$' &&
  want group-fail "$L" 0 '^mean ours d=4 ' &&
  want group-fail "$L" 1 '^failed arms: r1 ours d=4 rc=3; $'; then
  pass group-fail
fi

if [ "${DEPTH_QWEN3MOE_STUB_SHOW:-}" = 1 ]; then
  for L in "$tmp"/rotate.log "$tmp"/rotate-dry.log "$tmp"/mrs-noiter.log "$tmp"/warmup-rotate.log \
    "$tmp"/blocks.log "$tmp"/order-bad.log "$tmp"/warmup-bad.log "$tmp"/blocks-dry.log "$tmp"/ref-fail.log \
    "$tmp"/discard-fail.log "$tmp"/discard-nofit.log "$tmp"/warmup-fail.log "$tmp"/group-fail.log; do
    echo "--- ${L##*/}"
    cat "$L"
  done
fi
echo "depth-qwen3moe-stub: $n checks, $failed failed"
[ "$failed" = 0 ]
