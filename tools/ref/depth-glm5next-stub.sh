#!/usr/bin/env bash
# The depth-glm5next.sh stub test: the runner's arms with no lease, no card and no model. It copies the
# runner (DEPTH_GLM5NEXT_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# ref-paths.sh, models/glm5next.sh (the real profile: the arms run at its flags),
# timing-card.sh, lease-probe.sh, tdist.py, lcpp-fit.sh, cold-blocks.sh, gguf-ranges.py, records.py with
# generate_glm5next's checked-in schema, and a copy of lease.sh whose lease_take is replaced by a line
# that takes nothing; cards.sh there is depth-stub-cards.sh's, two made-up UUIDs. The profile keeps a
# caller's values, so the two PR trees' llama-bench and
# llama-server and generate_glm5next are stub scripts here, and MODEL a path nothing opens — or, for
# the preheat cases, gguf-ranges.py's two-shard fixture. Every case runs with BLOOMERY_PREHEAT=0 unless
# it sets 1. Nothing it starts loads a model or touches a card.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/depth-glm5next-stub.sh'
#   ... 'DEPTH_GLM5NEXT_RUNNER=<base copy> bash tools/ref/depth-glm5next-stub.sh'    # FAIL-first
#   ... 'DEPTH_GLM5NEXT_BASE=<base copy> bash tools/ref/depth-glm5next-stub.sh'      # with dry-same
#
# Runs on the box (bash 4 or later, GNU timeout). One line per check, `ok <name>` or `FAIL <name>:
# <why>` followed by the run's output; exit 0 iff none failed. DEPTH_GLM5NEXT_STUB_SHOW=1 prints every
# run's whole output after the checks.
#   fit          lcpp27754fit:6 lcpp27752:6 lcpp27752fit:6 lcpp27752ppfit:4 lcpp27754ppfit8:4
#                lcpp27754pp:4, one round, each arm after its own warm-up: the stub llama-bench echoes what it was
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
#   preheat      lcpp27752:6 (--n-cpu-moe 1 in its flags) and lcpp27752fit:6 on the fixture, GLM_PREHEAT_K
#                2, BLOOMERY_PREHEAT=1, one round, each arm after its warm-up: a `preheat` line before each
#                run, the warm-ups' included, K=1 (token_embd and layer 0's experts, 1416 B) for the hand-set arm
#                and K=2 (both layers', 2136 B) for the fit twin, and both `host` lines in [config].
#                Red on the runner before the preheat: no preheat line.
#   preheat-dry  the same arms' dry run: each arm's preheat K and bytes, both `host` lines, 3552 B a round.
#   preheat-refused  GLM_PREHEAT_K 3, past the fixture's block count: the fit arm's ranges refused before
#                the lease, rc 64, by name; -ot in the flags: refused by name, rc 64.
#   preheat-fail the preheat arms on a second fixture whose second shard the stub llama-bench removes when it
#                runs (STUB_BENCH_RM): the hand-set arm's warm-up and row read shard 1 alone (K=1) and
#                run; the fit twin's preheats (K=2) cannot open shard 2: its warm-up's and its row's FAIL
#                rows carry the preheat's rc 2 and its message, the runner rc 1. Red on a runner that loses the preheat's rc.
#   ours-timed   6 hot:6 through the stub generate_glm5next (needs the profile's prose ids, copied from
#                $BLOOMERY_DATA; skipped, and said so, without them), preheat on at GLM_PREHEAT_K 2 on the
#                fixture: each ours and hot row counts its faults from the `fed` line and prints the whole
#                process's beside it, and each arm is preheated at K=2; with STUB_GEN_NOFED=1 the rows
#                count the whole process and say there was no fed line. Red on the runner before the fed
#                mark.
#   cold-retry   6 through the stub generate_glm5next with faults in its timed window (STUB_GEN_FAULT):
#                faulting once, the row prints COLD, the arm runs once more and the second row is the ROW
#                in the mean; faulting every time, COLD then FAIL-cold, no mean, the failed list names it,
#                rc 1 (docs/fair-measure.md 2.3). Needs the prose ids, as ours-timed.
#   dry-same     DEPTH_GLM5NEXT_BASE set: the dry run of every arm kind the base runner knows, under the
#                base and under the runner tested, byte for byte less the lines this runner changed on
#                purpose: `[dry] preheat:`, `[dry] prompt:`, `[dry] WARMUP r0:` and the prompt id range of the
#                command lines and the server rows (skipped, and said so, without it).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNNER=${DEPTH_GLM5NEXT_RUNNER:-$ROOT/tools/ref/depth-glm5next.sh}
BASE=${DEPTH_GLM5NEXT_BASE:-}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/depth-glm5next-stub.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
T=$tmp/tree
mkdir -p "$T/tools/ref/models" "$T/tools/bloomery/schema" "$T/bin" "$T/pr27752" "$T/pr27754" "$T/data" \
  "$T/target/release" "$tmp/tmp"
cp "$RUNNER" "$T/tools/ref/depth-glm5next.sh"
cp "$ROOT/tools/ref/ref-paths.sh" "$ROOT/tools/ref/timing-card.sh" \
  "$ROOT/tools/ref/lease-probe.sh" "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" \
  "$ROOT/tools/ref/lcpp-fit.sh" "$ROOT/tools/ref/cold-blocks.sh" "$ROOT/tools/ref/gguf-ranges.py" "$T/tools/ref/"
cp "$ROOT/tools/ref/models/glm5next.sh" "$T/tools/ref/models/"
cp "$ROOT/tools/bloomery/records.py" "$T/tools/bloomery/"
# The tree's cards.sh only: this test writes its own nvidia-smi below.
STUB_CARDS_FILE_ONLY=1
# shellcheck source=tools/ref/depth-stub-cards.sh
. "$HERE/depth-stub-cards.sh"
unset STUB_CARDS_FILE_ONLY
cp "$ROOT/tools/bloomery/schema/generate_glm5next.jsonl" "$T/tools/bloomery/schema/"
# The preheat cases' model: gguf-ranges.py's fixture (block_count 2: token_embd and layer 0's experts in
# shard 1, layer 1's in shard 2).
FIX=$tmp/fix-00001-of-00002.gguf
python3 "$ROOT/tools/ref/gguf-ranges.py" fixture "$FIX" || { echo "FAIL setup: gguf-ranges.py fixture"; exit 1; }
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
# --n-cpu-moe STUB_BENCH_FAIL_K. It removes the file STUB_BENCH_RM names. --help lists --fit-target unless its tree is in STUB_BENCH_NO_FIT. Under
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
[ -z "${STUB_BENCH_RM:-}" ] || rm -f "$STUB_BENCH_RM"
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
# The stub generate_glm5next: --records-schema prints the checked-in schema; a run prints the records a
# timed run prints (the plan's card_experts nonzero, so a hot arm passes), the `fed` line left out under
# STUB_GEN_NOFED=1. Under STUB_GEN_FAULT=<file> it takes major faults after its `fed` line (the file
# written, its pages dropped, then read through a mapping) and times its prompt and steps at 0.01 ms,
# so the row is [cold]; with STUB_GEN_FAULT_ONCE=1 only its first run does.
G=$T/target/release/generate_glm5next
# shellcheck disable=SC2016 # ${1:-} is the stub's own argument
printf '#!/usr/bin/env bash\n[ "${1:-}" != --records-schema ] || exec cat %q\n' "$T/tools/bloomery/schema/generate_glm5next.jsonl" > "$G"
cat >> "$G" << 'EOF'
n=32 place=gate depth=0
while [ $# -gt 0 ]; do
  case $1 in
    --tokens) depth=$(echo "$2" | tr ',' '\n' | grep -c .); shift ;; -n) n=$2; shift ;; --place) place=$2; shift ;;
    --ctx | --warm) shift ;;
  esac
  shift
done
echo "plan place=$place card=A6000 ctx_max=2048 card_experts=2627 (39941832704 B) host_experts=9469 (145536581632 B) host_shadow=0 B n_l=67..68 on 39 layers card_budget=none hot_list=none"
echo "capture graph_nodes=1348"
[ -n "${STUB_GEN_NOFED:-}" ] || echo "fed ids=$depth first=[1, 2, 3, 4] last=[5, 6, 7, 8] depth_sequence_from=$depth"
pms=100.0000 sms=33.0000
if [ -n "${STUB_GEN_FAULT:-}" ] && { [ -z "${STUB_GEN_FAULT_ONCE:-}" ] || [ ! -e "$STUB_GEN_FAULT.done" ]; }; then
  touch "$STUB_GEN_FAULT.done"
  pms=0.0100 sms=0.0100
  # after the runner's mark reads the counter at the fed line
  sleep 0.5
  python3 - "$STUB_GEN_FAULT" << 'PY'
import mmap, os, sys
p = sys.argv[1]
with open(p, "wb") as f:
    f.write(os.urandom(1 << 20))
    f.flush()
    os.fsync(f.fileno())
fd = os.open(p, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
sum(m[i] for i in range(0, len(m), 4096))
PY
fi
echo "step 0 $((depth - 1)) 12 (the $depth fed steps in 0.1 s, runtime value)"
echo "time prompt n=$depth ms=$pms tok/s=$((depth * 10)).00 passes=$depth kind=steps"
for i in $(seq 1 $((n - 1))); do echo "step $i $((depth + i - 1)) $((1000 + i))"; done
for i in $(seq 1 $((n - 1))); do echo "time step $i ms=$sms"; done
echo "tokens [12, 1001]"
echo "SMOKE mode=graph place=$place prompt_tokens=$depth depth=$depth generated=$n warm=0 steps=$((n - 1)) p50_ms=$sms mean_ms=$sms tok/s(p50)=30.30"
EOF
chmod +x "$T/bin/"* "$T/pr27752/"* "$T/pr27754/"* "$G"

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
    BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 BLOOMERY_CPU_BUSY_COMMS=none BLOOMERY_TIMING_GPU= BLOOMERY_PREHEAT=0 "${e[@]}" \
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
  want fit "$L" 2 '^    lcpp27752fit table \| pr27752 ngl= k= fa=on fitt=1024 v=1 ub= tf32=unset \| 1 \| tg4 @ d6 \|' &&
  want fit "$L" 2 '^    lcpp27752ppfit table \| pr27752 ngl= k= fa=on fitt=1024 v=1 ub= tf32=unset \| 1 \| pp4 \|' &&
  want fit "$L" 2 '^    lcpp27754ppfit8 table \| pr27754 ngl= k= fa=off fitt=1024 v=1 ub=8 tf32=0 \| 1 \| pp4 \|' &&
  want fit "$L" 2 '^    lcpp27752 table \| pr27752 ngl=999 k=36 fa=on fitt= v= ub= tf32=unset \|' &&
  want fit "$L" 2 '^    lcpp27754pp table \| pr27754 ngl=999 k=36 fa=off fitt= v= ub= tf32=0 \|' &&
  want fit "$L" 8 '^    lcpp2775[24](pp)?fit8? fit load_tensors: offloaded 3/3 layers to GPU$' &&
  want fit "$L" 8 '^    lcpp2775[24](pp)?fit8? fit load_tensors: +CPU_Mapped model buffer size = +2.00 MiB$' &&
  want fit "$L" 6 '^WARMUP r0 ' &&
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

PH_ENV=(BLOOMERY_PREHEAT=1 BLOOMERY_REF_MODEL="$FIX" GLM_PREHEAT_K=2
  LCPP27752_GPU_FLAGS='-ngl 999 --n-cpu-moe 1 -fa on -t 32 -nopo 1')
L=$tmp/preheat.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "${PH_ENV[@]}" -- lcpp27752:6 lcpp27752fit:6
if [ "$RC" != 0 ]; then
  fail preheat "rc $RC, want 0" "$L"
elif want preheat "$L" 2 '^preheat lcpp27752 K=1 bytes=1416 s=[0-9.]+ gbps=' &&
  want preheat "$L" 2 '^preheat lcpp27752fit K=2 bytes=2136 s=[0-9.]+ gbps=' &&
  want preheat "$L" 4 '^preheat ' &&
  want preheat "$L" 1 '^\[config\] preheat: host K=1 layers=2 tensors=3 bytes=1416 ranges=2 shards=2 exps_first=1320 exps_last=1320 dense=0$' &&
  want preheat "$L" 1 '^\[config\] preheat: host K=2 layers=2 tensors=5 bytes=2136 ' &&
  want preheat "$L" 1 '^WARMUP r0 lcpp27752 d=6 ' &&
  want preheat "$L" 1 '^ROW r1 lcpp27752fit d=6 ' &&
  { [ "$(grep -A1 '^preheat lcpp27752fit ' "$L" | tail -n 1 | cut -c1-17)" = '--- witness pre R' ] ||
    { fail preheat "the fit arm's preheat line is not right before its witness block" "$L"; false; }; }; then
  pass preheat
fi

L=$tmp/preheat-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 "${PH_ENV[@]}" -- lcpp27752:6 lcpp27752fit:6
if [ "$RC" != 0 ]; then
  fail preheat-dry "rc $RC, want 0" "$L"
elif want preheat-dry "$L" 1 '^\[dry\]     preheat K=1: 1416 B \(0.0 GB\), 0 s if all of it is cold at 1.33 GB/s$' &&
  want preheat-dry "$L" 1 '^\[dry\]     preheat K=2: 2136 B ' &&
  want preheat-dry "$L" 2 '^\[dry\] preheat: host K=[12] ' &&
  want preheat-dry "$L" 1 '^\[dry\] preheat: 3552 B a round over the GGUF arms, ' &&
  want preheat-dry "$L" 0 '^preheat '; then
  pass preheat-dry
fi

L=$tmp/preheat-refused.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "${PH_ENV[@]}" GLM_PREHEAT_K=3 -- lcpp27752:6 lcpp27752fit:6
L2=$tmp/preheat-ot.log
RC1=$RC
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 "${PH_ENV[@]}" LCPP27752_GPU_FLAGS='-ngl 999 --n-cpu-moe 1 -ot exps=CPU' -- lcpp27752:6
if [ "$RC1" != 64 ]; then
  fail preheat-refused "K past the block count: rc $RC1, want 64" "$L"
elif [ "$RC" != 64 ]; then
  fail preheat-refused "-ot: rc $RC, want 64" "$L2"
elif want preheat-refused "$L" 1 "^depth-glm5next.sh: arm 'lcpp27752fit:6': no preheat ranges for K=3 \(tools/ref/gguf-ranges.py rc 64\)$" &&
  want preheat-refused "$L" 1 '^gguf-ranges.py: --n-cpu-moe 3 outside 0..2 ' &&
  want preheat-refused "$L" 0 '^(ROW|WARMUP|FAIL|preheat) ' &&
  want preheat-refused "$L2" 1 "^depth-glm5next.sh: arm 'lcpp27752:6': its flags carry -ot, a host set the preheat does not model; set BLOOMERY_PREHEAT=0 " &&
  want preheat-refused "$L2" 0 '^(ROW|WARMUP|FAIL|preheat) '; then
  pass preheat-refused
fi

FIX2=$tmp/fix2-00001-of-00002.gguf
python3 "$ROOT/tools/ref/gguf-ranges.py" fixture "$FIX2" || { echo "FAIL setup: gguf-ranges.py fixture"; exit 1; }
L=$tmp/preheat-fail.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "${PH_ENV[@]}" BLOOMERY_REF_MODEL="$FIX2" STUB_BENCH_RM="${FIX2%-00001-of-00002.gguf}-00002-of-00002.gguf" -- lcpp27752:6 lcpp27752fit:6
if [ "$RC" != 1 ]; then
  fail preheat-fail "rc $RC, want 1" "$L"
elif want preheat-fail "$L" 1 '^FAIL r1 lcpp27752fit d=6 rc=2 \| its preheat failed \(rc 2\): gguf-ranges.py: .*fix2-00002-of-00002.gguf: No such file or directory \| full output: ' &&
  want preheat-fail "$L" 1 '^FAIL r0 lcpp27752fit d=6 rc=2 \| its preheat failed \(rc 2\): ' &&
  want preheat-fail "$L" 2 '^preheat lcpp27752 K=1 ' &&
  want preheat-fail "$L" 1 '^ROW r1 lcpp27752 d=6 ' &&
  want preheat-fail "$L" 1 '^failed arms: r0:lcpp27752fit@d=6 r1:lcpp27752fit@d=6$'; then
  pass preheat-fail
fi

PROSE_SRC=${BLOOMERY_DATA:-/root/bloomery-data}/glm5next/corpus-prose.ids
if [ -f "$PROSE_SRC" ]; then
  mkdir -p "$T/data/glm5next" "$T/data/router"
  cp "$PROSE_SRC" "$T/data/glm5next/"
  echo "stub hot list" > "$T/data/router/glm5next-hotlist.txt"
  GPU_A=$STUB_GPU_A6000
  L=$tmp/ours-timed.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" "${PH_ENV[@]}" -- 6 hot:6
  L2=$tmp/ours-nofed.log
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_NOFED=1 GLM_PROSE_FROM=0 -- 6 hot:6
  if [ "$RC1" != 0 ]; then
    fail ours-timed "rc $RC1, want 0" "$L"
  elif [ "$RC" != 0 ]; then
    fail ours-timed "no fed line: rc $RC, want 0" "$L2"
  elif want ours-timed "$L" 2 '^ROW r1 (ours|hot) d=6 n=4 ctx=2048 \| .*\| prompt ids 50000..50005 \| .*\| majflt [0-9]+ \(timed, from the fed line; whole process [0-9]+\) <= [0-9.]+ % of the timed window \| wall [0-9]+s$' &&
    want ours-timed "$L" 2 '^WARMUP r0 (ours|hot) d=6 .*\| majflt [0-9]+ \(timed, from the fed line; ' &&
    want ours-timed "$L" 4 '^preheat (ours|hot) K=2 bytes=2136 ' &&
    want ours-timed "$L" 1 '^\[config\] prompt: GLM_PROSE ids from index 50000; the hot list.s trace: ids 0..49999$' &&
    want ours-timed "$L" 0 '\(whole process\)' &&
    want ours-timed "$L2" 2 '^ROW r1 (ours|hot) d=6 .*\| majflt [0-9]+ \(whole process: no fed line\) <= ' &&
    want ours-timed "$L2" 1 '^ROW r1 hot d=6 .*\| prompt ids 0..5 \| .* \[in-trace\]$' &&
    want ours-timed "$L2" 1 '^ROW r1 ours d=6 .*\| prompt ids 0..5 \| .*s$' &&
    want ours-timed "$L2" 0 '^preheat '; then
    pass ours-timed
  fi

  # The cold rule: a [cold] row runs once more; a second [cold] is FAIL-cold.
  FAULTF=$ROOT/target/depth-glm5next-stub-fault.$$.bin
  mkdir -p "$ROOT/target"
  L=$tmp/cold-retry.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_FAULT="$FAULTF" STUB_GEN_FAULT_ONCE=1 -- 6
  L2=$tmp/cold-fail.log
  RC1=$RC
  rm -f "$FAULTF.done"
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_FAULT="$FAULTF" -- 6
  rm -f "$FAULTF" "$FAULTF.done"
  if [ "$RC1" != 0 ]; then
    fail cold-retry "rc $RC1, want 0" "$L"
  elif [ "$RC" != 1 ]; then
    fail cold-retry "cold twice: rc $RC, want 1" "$L2"
  elif want cold-retry "$L" 1 '^COLD r1 ours d=6 .*\| majflt [1-9][0-9]* \(timed, from the fed line; .* \[cold\]$' &&
    want cold-retry "$L" 1 '^\[cold\] r1 6 read \[cold\]: its arm runs once more ' &&
    want cold-retry "$L" 1 '^ROW r1 ours d=6 .*s$' &&
    want cold-retry "$L" 1 '^mean ours +6 +[0-9.]+ tok/s +\[.*\(n=1\)$' &&
    want cold-retry "$L" 1 'cold re-runs 1, FAIL-cold 0$' &&
    want cold-retry "$L2" 1 '^COLD r1 ours d=6 .* \[cold\]$' &&
    want cold-retry "$L2" 1 '^FAIL-cold r1 ours d=6 .* \[cold\]$' &&
    want cold-retry "$L2" 0 '^(ROW|mean) ' &&
    want cold-retry "$L2" 1 '^failed arms: r1:ours@d=6\(cold\)$'; then
    pass cold-retry
  fi
else
  echo "skip ours-timed and cold-retry: no prose ids at $PROSE_SRC (the profile pins their sha256)"
fi

# same_view: a dry run less the lines this runner changes on purpose (dry-same).
same_view() {
  grep -vE '^\[dry\] (preheat|prompt|WARMUP r0):' |
    sed -E 's/--tokens <[^>]*>/--tokens <prompt>/; s/row "POST \/completion: [^,]*,/row "POST \/completion: <prompt>,/'
}
if [ -n "$BASE" ]; then
  SAME_ARMS=(6 hot:6 lcpp27752:6 lcpp27754:6 lcpp27752pp:4 lcpp27754pp:4 lcpp27752pp8:4 lcpp27754pp4096:4 lcpp27752srv:6 lcpp27754srv:6 lcpp27752mtp:6 lcpp27754mtp:6 exl3:256 exl3pp:256)
  cp "$BASE" "$T/tools/ref/depth-glm5next-base.sh"
  L=$tmp/dry-same-base.log L2=$tmp/dry-same.log
  RUNNER_FILE=tools/ref/depth-glm5next-base.sh stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- "${SAME_ARMS[@]}"
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- "${SAME_ARMS[@]}"
  if [ "$RC1" != 0 ] || [ "$RC" != 0 ]; then
    fail dry-same "rc $RC1 (base) and $RC (tested), want 0 and 0" "$L2"
  elif ! diff <(same_view < "$L") <(same_view < "$L2") > "$tmp/dry-same.diff"; then
    fail dry-same "the dry runs differ, less the lines changed on purpose (base <, tested >)" "$tmp/dry-same.diff"
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
