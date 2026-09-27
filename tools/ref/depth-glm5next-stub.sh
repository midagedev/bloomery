#!/usr/bin/env bash
# The depth-glm5next.sh stub test: the runner's reference arms with no lease, no card and no model. It
# copies the runner (DEPTH_GLM5NEXT_RUNNER, default this tree's) into a fresh temporary tree beside this
# tree's ref-paths.sh, models/glm5next.sh (the real profile: the arms run at its flags), cards.sh,
# timing-card.sh, lease-probe.sh, tdist.py, lcpp-fit.sh and a copy of lease.sh whose lease_take is
# replaced by a line that takes nothing. The profile keeps a caller's values, so the two PR trees'
# llama-bench and llama-server are stub scripts here, and MODEL a path nothing opens. Nothing it starts
# loads a model or touches a card.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/depth-glm5next-stub.sh'
#   ... 'DEPTH_GLM5NEXT_RUNNER=<base copy> bash tools/ref/depth-glm5next-stub.sh'    # FAIL-first
#   ... 'DEPTH_GLM5NEXT_BASE=<base copy> bash tools/ref/depth-glm5next-stub.sh'      # with dry-same
#
# Runs on the box (bash 4 or later, GNU timeout). One line per check, `ok <name>` or `FAIL <name>:
# <why>` followed by the run's output; exit 0 iff none failed. DEPTH_GLM5NEXT_STUB_SHOW=1 prints every
# run's whole output after the checks.
#   fit          lcpp27754fit:6 lcpp27752:6 lcpp27752fit:6 lcpp27752ppfit:4 lcpp27754ppfit8:4
#                lcpp27754pp:4, one round after the warm-up: the stub llama-bench echoes what it was
#                given in its table's model column. The fit arms' show no -ngl and no --n-cpu-moe, and
#                -fitt 1024 -v; #27754's keep NVIDIA_TF32_OVERRIDE=0 and -fa off, #27752's -fa on. Under
#                -fitt -v the stub prints two loads, the last with two overrides; every fit row, the
#                warm-up's included, carries the `fit` column read from that last load and echoes its
#                lines, and the hand-set rows carry none (a hand-set row after a fit row included).
#                Red on the runner before the fit arms: arm usage, rc 64.
#   fit-fail     a llama-bench whose fit fails (common_fit_params' warning) and measures anyway: each fit
#                arm is a FAIL row naming it (rc=0), the hand-set arm a ROW, the runner rc 1.
#   fit-exit     the same failure, then an exit 1 (the load at -ngl -1 out of card memory): the fit arm's
#                FAIL row names the exit and the fit; a hand-set arm refused at its --n-cpu-moe is the
#                plain `exited 1` row it was.
#   fit-nobench  #27754's llama-bench lists no --fit-target: lcpp27754fit is refused by name, rc 64,
#                before any row; lcpp27752fit beside a hand-set lcpp27754 arm runs (the probe is per
#                branch).
#   fit-flags    LCPP27752_GPU_FLAGS already carrying -fitt: the fit arm refused by name, rc 64.
#   fit-dry      the dry run: each fit arm's command line (the profile's flags less -ngl 999 and
#                --n-cpu-moe 36, then -fitt 1024 -v; #27754's env and -fa off kept) and what it dropped.
#   dry-same     DEPTH_GLM5NEXT_BASE set: the dry run of every arm kind the base runner knows, under the
#                base and under the runner tested, byte for byte (skipped, and said so, without it).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNNER=${DEPTH_GLM5NEXT_RUNNER:-$ROOT/tools/ref/depth-glm5next.sh}
BASE=${DEPTH_GLM5NEXT_BASE:-}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/depth-glm5next-stub.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
T=$tmp/tree
mkdir -p "$T/tools/ref/models" "$T/tools/bloomery" "$T/bin" "$T/pr27752" "$T/pr27754" "$T/data" "$tmp/tmp"
cp "$RUNNER" "$T/tools/ref/depth-glm5next.sh"
cp "$ROOT/tools/ref/ref-paths.sh" "$ROOT/tools/ref/timing-card.sh" "$ROOT/tools/ref/cards.sh" \
  "$ROOT/tools/ref/lease-probe.sh" "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" \
  "$ROOT/tools/ref/lcpp-fit.sh" "$T/tools/ref/"
cp "$ROOT/tools/ref/models/glm5next.sh" "$T/tools/ref/models/"
cp "$ROOT/tools/bloomery/records.py" "$T/tools/bloomery/"
echo 'lease_take() { echo "[stub] no lease: the stub test'"'"'s copy of lease.sh takes nothing"; }' >> "$T/tools/ref/lease.sh"
touch "$T/Cargo.toml"
cat > "$T/bin/nvidia-smi" << 'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=name\ *) echo "NVIDIA RTX A6000 (stub)" ;;
  *--query-compute-apps*) ;;
  *) echo "stub, stub, stub, stub, stub, stub, stub, stub" ;;
esac
EOF
# The stub llama-bench, one copy a PR tree (the tree is its directory's name). Its table's model column
# echoes what it was given: -ngl, --n-cpu-moe, -fa, -fitt, -v, -ub and NVIDIA_TF32_OVERRIDE. It refuses
# --n-cpu-moe STUB_BENCH_FAIL_K. --help lists --fit-target unless its tree is in STUB_BENCH_NO_FIT. Under
# -fitt it prints common_fit_params' failure warning when STUB_BENCH_FIT_FAIL is set, and then exits 1
# when STUB_BENCH_FIT_EXIT is set too; under -fitt -v it prints two model loads, the fit's measuring
# one and the real one with two expert tensors of blk 1 overridden to the host.
cat > "$T/pr27752/llama-bench" << 'EOF'
#!/usr/bin/env bash
tree=$(basename "$(dirname "$0")") ngl='' k='' fa='' fitt='' verb='' ub='' p='' n='' d=''
while [ $# -gt 0 ]; do
  case $1 in
    -ngl) ngl=$2; shift ;; --n-cpu-moe) k=$2; shift ;; -fa) fa=$2; shift ;; -fitt) fitt=$2; shift ;;
    -ub) ub=$2; shift ;; -p) p=$2; shift ;; -n) n=$2; shift ;; -d) d=$2; shift ;; -v) verb=1 ;;
    -h | --help)
      echo "usage: llama-bench [options]"
      case " ${STUB_BENCH_NO_FIT:-} " in
        *" $tree "*) ;;
        *) echo "  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)" ;;
      esac
      exit 0
      ;;
  esac
  shift
done
if [ -n "$fitt" ]; then
  if [ -n "${STUB_BENCH_FIT_FAIL:-}" ]; then
    echo "common_fit_params: failed to fit params to free device memory: stub" >&2
    if [ -n "${STUB_BENCH_FIT_EXIT:-}" ]; then
      echo "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 170000.00 MiB on device 0: cudaMalloc failed: out of memory (stub)" >&2
      exit 1
    fi
  fi
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
if [ -n "$k" ] && [ "$k" = "${STUB_BENCH_FAIL_K:-none}" ]; then
  echo "llama_init_from_model: failed to create context (stub: --n-cpu-moe $k)" >&2
  exit 1
fi
if [ "${n:-0}" = 0 ]; then label="pp$p"; else label="tg$n @ d$d"; fi
echo "| model | size | test | t/s |"
echo "| --- | ---: | ---: | ---: |"
echo "| $tree ngl=$ngl k=$k fa=$fa fitt=$fitt v=$verb ub=$ub tf32=${NVIDIA_TF32_OVERRIDE:-unset} | 1 | $label | 20.00 ± 0.01 |"
echo "build: stub (0)"
EOF
cp "$T/pr27752/llama-bench" "$T/pr27754/llama-bench"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/pr27752/llama-server"
cp "$T/pr27752/llama-server" "$T/pr27754/llama-server"
chmod +x "$T/bin/"* "$T/pr27752/"* "$T/pr27754/"*

n=0 failed=0
pass() { n=$((n + 1)); echo "ok $1"; }
fail() {
  n=$((n + 1)) failed=$((failed + 1))
  echo "FAIL $1: $2"
  [ -z "${3:-}" ] || sed 's/^/    | /' "$3"
}
# stub_run <log> <env…> -- <arms…>: the runner in the stub tree, on the real profile with the stub
# engines; its rc into RC. RUNNER_FILE (default the runner under test) picks the script.
stub_run() {
  local log=$1 e=()
  shift
  while [ "$1" != -- ]; do e+=("$1"); shift; done
  shift
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" BLOOMERY_MODEL=glm5next BLOOMERY_REF_MODEL_PROFILE=glm5next \
    BLOOMERY_REF_MODEL="$tmp/m-00001-of-00006.gguf" BLOOMERY_DATA="$T/data" \
    LCPP27752="$T/pr27752" LCPP27752BIN="$T/pr27752/llama-bench" LCPP27752SRV="$T/pr27752/llama-server" \
    LCPP27754="$T/pr27754" LCPP27754BIN="$T/pr27754/llama-bench" LCPP27754SRV="$T/pr27754/llama-server" \
    EXL3="$T/exl3" EXL3_PY="$T/exl3/python" EXL3_MODEL="$T/exl3-model" EXL3_WIKITEXT="$T/wiki.test.raw" \
    BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 BLOOMERY_CPU_BUSY_COMMS=none BLOOMERY_TIMING_GPU= "${e[@]}" \
    bash "${RUNNER_FILE:-tools/ref/depth-glm5next.sh}" "$@") > "$log" 2>&1
  RC=$?
}
# want <name> <log> <count> <pattern>: the log holds exactly <count> lines matching grep -E <pattern>.
want() {
  local c
  c=$(grep -cE -- "$4" "$2")
  [ "$c" = "$3" ] || { fail "$1" "$c lines match /$4/, want $3" "$2"; return 1; }
}

FIT_ROW='A6000 \(stub\) \| fit offloaded 3/3, overridden CPU:2 in blk 1-1 \(blk 1: 2\), buffers CUDA0 10.00 CPU_Mapped 2.00 MiB \|'
L=$tmp/fit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- lcpp27754fit:6 lcpp27752:6 lcpp27752fit:6 lcpp27752ppfit:4 lcpp27754ppfit8:4 lcpp27754pp:4
if [ "$RC" != 0 ]; then
  fail fit "rc $RC, want 0" "$L"
elif want fit "$L" 1 "^WARMUP r0 lcpp27754fit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, $FIT_ROW build stub \(0\) " &&
  want fit "$L" 1 "^ROW r1 lcpp27754fit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, $FIT_ROW build " &&
  want fit "$L" 1 "^ROW r1 lcpp27752fit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, $FIT_ROW build " &&
  want fit "$L" 1 "^ROW r1 lcpp27752ppfit p=4 n=0 \| tok/s\(pp\) 20.00 @ n=0, prompt 4, $FIT_ROW ub 512 b 2048 \(llama-bench defaults\) \| build " &&
  want fit "$L" 1 "^ROW r1 lcpp27754ppfit8 p=4 n=0 \| tok/s\(pp\) 20.00 @ n=0, prompt 4, $FIT_ROW ub 8 b 2048 \(the arm.s lever\) \| build " &&
  want fit "$L" 1 '^ROW r1 lcpp27752 d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) \| build ' &&
  want fit "$L" 1 '^ROW r1 lcpp27754pp p=4 n=0 \| tok/s\(pp\) 20.00 @ n=0, prompt 4, A6000 \(stub\) \| ub 512 ' &&
  want fit "$L" 0 '^(ROW|WARMUP) r[01] lcpp2775[24](pp)? .*\| fit ' &&
  want fit "$L" 2 '^    lcpp27754fit table \| pr27754 ngl= k= fa=off fitt=1024 v=1 ub= tf32=0 \| 1 \| tg4 @ d6 \|' &&
  want fit "$L" 1 '^    lcpp27752fit table \| pr27752 ngl= k= fa=on fitt=1024 v=1 ub= tf32=unset \| 1 \| tg4 @ d6 \|' &&
  want fit "$L" 1 '^    lcpp27752ppfit table \| pr27752 ngl= k= fa=on fitt=1024 v=1 ub= tf32=unset \| 1 \| pp4 \|' &&
  want fit "$L" 1 '^    lcpp27754ppfit8 table \| pr27754 ngl= k= fa=off fitt=1024 v=1 ub=8 tf32=0 \| 1 \| pp4 \|' &&
  want fit "$L" 1 '^    lcpp27752 table \| pr27752 ngl=999 k=36 fa=on fitt= v= ub= tf32=unset \|' &&
  want fit "$L" 1 '^    lcpp27754pp table \| pr27754 ngl=999 k=36 fa=off fitt= v= ub= tf32=0 \|' &&
  want fit "$L" 5 '^    lcpp2775[24](pp)?fit8? fit load_tensors: offloaded 3/3 layers to GPU$' &&
  want fit "$L" 5 '^    lcpp2775[24](pp)?fit8? fit load_tensors: +CPU_Mapped model buffer size = +2.00 MiB$' &&
  want fit "$L" 1 '^\[config\] lcpp27752fit: flags=-fa on -t 32 -nopo 1 -fitt 1024 -v \(llama-bench.s fit places the model; dropped: -ngl 999 --n-cpu-moe 36;' &&
  want fit "$L" 1 '^\[config\] lcpp27754fit: flags=-fa off -t 32 -nopo 1 -fitt 1024 -v \(' &&
  want fit "$L" 1 '^mean lcpp27752fit +6 ' &&
  want fit "$L" 1 '^mean lcpp27754ppfit8 +4 ' &&
  want fit "$L" 0 '^failed arms'; then
  pass fit
fi

L=$tmp/fit-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_BENCH_FIT_FAIL=1 -- lcpp27752fit:6 lcpp27754ppfit:4 lcpp27752:6
if [ "$RC" != 1 ]; then
  fail fit-fail "rc $RC, want 1" "$L"
elif want fit-fail "$L" 1 "^FAIL r1 lcpp27752fit d=6 rc=0 \| llama-bench.s fit failed, and llama-bench loads without it: common_fit_params: failed to fit params to free device memory: stub \| full output: " &&
  want fit-fail "$L" 1 "^FAIL r1 lcpp27754ppfit p=4 rc=0 \| llama-bench.s fit failed, " &&
  want fit-fail "$L" 1 '^ROW r1 lcpp27752 d=6 ' &&
  want fit-fail "$L" 0 '^ROW r1 lcpp2775[24](pp)?fit ' &&
  want fit-fail "$L" 1 '^failed arms: r1:lcpp27752fit@d=6 r1:lcpp27754ppfit@p=4$'; then
  pass fit-fail
fi

L=$tmp/fit-exit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_BENCH_FIT_FAIL=1 STUB_BENCH_FIT_EXIT=1 STUB_BENCH_FAIL_K=36 -- lcpp27752fit:6 lcpp27754:6
if [ "$RC" != 1 ]; then
  fail fit-exit "rc $RC, want 1" "$L"
elif want fit-exit "$L" 1 "^FAIL r1 lcpp27752fit d=6 rc=1 \| exited 1; llama-bench.s fit failed, and llama-bench loads without it: common_fit_params: " &&
  want fit-exit "$L" 1 '^FAIL r1 lcpp27754 d=6 rc=1 \| exited 1 \| full output: ' &&
  want fit-exit "$L" 0 '^ROW '; then
  pass fit-exit
fi

L=$tmp/fit-nobench.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_BENCH_NO_FIT=pr27754 -- lcpp27752:6 lcpp27754fit:6
L2=$tmp/fit-nobench-other.log
RC1=$RC
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 STUB_BENCH_NO_FIT=pr27754 -- lcpp27752fit:6 lcpp27754:6
if [ "$RC1" != 64 ]; then
  fail fit-nobench "rc $RC1, want 64" "$L"
elif [ "$RC" != 0 ]; then
  fail fit-nobench "the other branch's fit arm: rc $RC, want 0" "$L2"
elif want fit-nobench "$L" 1 "^depth-glm5next.sh: the lcpp27754fit/lcpp27754ppfit arms need llama-bench.s fit: .*/pr27754/llama-bench --help lists no -fitt/--fit-target: this llama-bench has no fit \(tree .*/pr27754\)$" &&
  want fit-nobench "$L" 0 '^(ROW|WARMUP|FAIL) ' &&
  want fit-nobench "$L2" 1 '^ROW r1 lcpp27752fit d=6 .*\| fit offloaded ' &&
  want fit-nobench "$L2" 1 '^ROW r1 lcpp27754 d=6 '; then
  pass fit-nobench
fi

L=$tmp/fit-flags.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 LCPP27752_GPU_FLAGS='-ngl 999 --n-cpu-moe 36 -fitt 512' -- lcpp27752fit:6
if [ "$RC" != 64 ]; then
  fail fit-flags "rc $RC, want 64" "$L"
elif want fit-flags "$L" 1 "^depth-glm5next.sh: arm 'lcpp27752fit:6' is .* — the profile's flags already carry -fitt \(-ngl 999 --n-cpu-moe 36 -fitt 512\); the fit arm would add a second value$" &&
  want fit-flags "$L" 0 '^(ROW|WARMUP|FAIL) '; then
  pass fit-flags
fi

L=$tmp/fit-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- lcpp27752:6 lcpp27752fit:6 lcpp27754fit:6 lcpp27754ppfit8:4
if [ "$RC" != 0 ]; then
  fail fit-dry "rc $RC, want 0" "$L"
elif want fit-dry "$L" 1 "^\[dry\] lcpp27752fit:6: row \"tg4 @ d6\", placement: llama-bench.s fit at -fitt 1024 MiB \(dropped: -ngl 999 --n-cpu-moe 36\), -v for the fit column$" &&
  want fit-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env [^ ]*/pr27752/llama-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -fa on -t 32 -nopo 1 -fitt 1024 -v $" &&
  want fit-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env NVIDIA_TF32_OVERRIDE=0 [^ ]*/pr27754/llama-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -fa off -t 32 -nopo 1 -fitt 1024 -v $" &&
  want fit-dry "$L" 1 "^\[dry\] lcpp27754ppfit8:4: row \"pp4\", ub 8 b 2048 \(the arm.s lever\), placement: llama-bench.s fit at -fitt 1024 MiB \(dropped: -ngl 999 --n-cpu-moe 36\), " &&
  want fit-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env NVIDIA_TF32_OVERRIDE=0 [^ ]*/pr27754/llama-bench -m [^ ]* -p 4 -n 0 -r 1 -ub 8 -b 2048 -fa off -t 32 -nopo 1 -fitt 1024 -v $" &&
  want fit-dry "$L" 1 "^\[dry\] lcpp27752:6: row \"tg4 @ d6\"$" &&
  want fit-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env [^ ]*/pr27752/llama-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -ngl 999 --n-cpu-moe 36 -fa on -t 32 -nopo 1 $" &&
  want fit-dry "$L" 1 '^\[dry\] round 1 gguf order: lcpp27752:6 lcpp27752fit:6 lcpp27754fit:6 lcpp27754ppfit8:4 $'; then
  pass fit-dry
fi

if [ -n "$BASE" ]; then
  SAME_ARMS=(6 hot:6 lcpp27752:6 lcpp27754:6 lcpp27752pp:4 lcpp27754pp:4 lcpp27752pp8:4 lcpp27754pp4096:4 lcpp27752srv:6 lcpp27754srv:6 lcpp27752mtp:6 lcpp27754mtp:6 exl3:256 exl3pp:256)
  cp "$BASE" "$T/tools/ref/depth-glm5next-base.sh"
  L=$tmp/dry-same-base.log L2=$tmp/dry-same.log
  RUNNER_FILE=tools/ref/depth-glm5next-base.sh stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- "${SAME_ARMS[@]}"
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- "${SAME_ARMS[@]}"
  if [ "$RC1" != 0 ] || [ "$RC" != 0 ]; then
    fail dry-same "rc $RC1 (base) and $RC (tested), want 0 and 0" "$L2"
  elif ! diff "$L" "$L2" > "$tmp/dry-same.diff"; then
    fail dry-same "the dry runs differ (base <, tested >)" "$tmp/dry-same.diff"
  elif want dry-same "$L2" 14 '^\[dry\]     '; then
    pass dry-same
  fi
else
  echo "skip dry-same: DEPTH_GLM5NEXT_BASE is unset (a copy of the base runner to compare the dry run with)"
fi

if [ "${DEPTH_GLM5NEXT_STUB_SHOW:-}" = 1 ]; then
  for L in "$tmp"/*.log; do
    echo "--- ${L##*/}"
    cat "$L"
  done
fi
echo "depth-glm5next-stub: $n checks, $failed failed"
[ "$failed" = 0 ]
