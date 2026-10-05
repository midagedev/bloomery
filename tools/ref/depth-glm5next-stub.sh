#!/usr/bin/env bash
# The depth-glm5next.sh stub test: the runner's arms with no lease, no card and no model. It copies the
# runner (DEPTH_GLM5NEXT_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# ref-paths.sh, models/glm5next.sh (the real profile: the arms run at its flags),
# timing-card.sh, lease-probe.sh, tdist.py, lcpp-fit.sh, lcpp-warm.sh, cold-blocks.sh, lever-arms.sh, arm-place.sh, slots-arm.sh,
# gguf-ranges.py,
# records.py with generate_glm5next's checked-in schema, the lever registry, and a copy of lease.sh whose lease_take is replaced by a line
# that takes nothing; cards.sh there is depth-stub-cards.sh's, two made-up UUIDs. The fault counter every
# copy reads (majflt_now, majflt_mark, lcpp_srv_majflt) is a file the stub engines add to when a case
# makes them fault, so a row is cold exactly where a case says — with the real /proc/vmstat counters the
# box's ambient page faults (the lease is what keeps them to the arm's process) would tag random arms
# [cold] on the stubs' ~0.1 s windows, a different case each run. The profile keeps a
# caller's values, so the two PR trees' llama-bench and generate_glm5next are stub scripts here, their
# llama-server tools/ref/stub-llama-server.py, and MODEL a path nothing opens — or, for
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
#   ours-timed   6 through the stub generate_glm5next (needs the profile's prose ids, copied from
#                $BLOOMERY_DATA; skipped, and said so, without them), preheat on at GLM_PREHEAT_K 2 on the
#                fixture: each ours row counts its faults from the `fed` line and prints the whole
#                process's beside it, and each arm is preheated at K=2; with STUB_GEN_NOFED=1 the rows
#                count the whole process and say there was no fed line. Red on the runner before the fed
#                mark.
#   cold-retry   6 through the stub generate_glm5next with faults in its timed window (STUB_GEN_FAULT):
#                faulting once, the row prints COLD, the arm runs once more and the second row is the ROW
#                in the mean; faulting every time, COLD then FAIL-cold, no mean, the failed list names it,
#                rc 1 (docs/fair-measure.md 2.3). Needs the prose ids, as ours-timed.
#   srv          6 lcpp27754srv:6 lcpp27752mtp:6 lcpp27754srvpp:6, one round (needs the prose ids, as
#                ours-timed): each server arm's command line is the branch's bench flags in the server's
#                spellings with -fit off, the MTP words after them, -np 1 -ctxcp 0 --cache-ram 0 and -c 2048
#                (ours' --ctx), #27754's under NVIDIA_TF32_OVERRIDE=0 (its srv and srvpp arms, no ubatch lever,
#                the same line); two requests of the prose ids a server
#                arm (the warm-up and the timed); no WARMUP r0 process for a server arm; the MTP row's draft
#                fields and acceptance line; the decode rows' prompt rate in the prefill means; each server
#                row's xcheck `same`. Red on the runner before lcpp-warm.sh: no ctxcp, no warm-up request.
#   srv-xcheck   the same with ours' token 0 another id (STUB_GEN_TOKEN0=7): the FAIL xcheck line in the
#                failed arms, the server label dropped from the means, rc 1; the server's second id another
#                (STUB_SRV_TOKEN1=7): a [xcheck-tail] line, the row kept in the means, rc 0.
#   srv-ctx      a server whose log names n_ctx 4096 (STUB_SRV_NCTX): its FAIL row names both contexts;
#                an ours load record naming ctx 4096 (STUB_GEN_LOAD_CTX): its FAIL row names both.
#   srv-flags    lcpp27754srv+t16+k40:6: -t 16 and --n-cpu-moe 40 on its command line, the label carrying
#                them; lcpp27754srv+x1:6 and lcpp27754+t16:6 refused by name, rc 64.
#   pair         BLOOMERY_GEN_PAIR=1: `--pair` at the end of our arm's dry command line; BLOOMERY_GEN_PAIR=yes:
#                refused by name, rc 64, before any row.
#   srv-past     lcpp27754srv:2046 at C 2048, N 4: refused before the lease by name, rc 64.
#   srv-fail     an MTP server that drafts nothing (STUB_SRV_NODRAFT): its FAIL row; a server that exits
#                before /health (STUB_SRV_EXIT): its FAIL row names the -c; a server whose --help lists no
#                --spec-type: the MTP arm refused by name, rc 64.
#   srv-dry      the dry server command lines and the `[dry] server arms:` line. Every srv case but srv-past
#                and the refusals of srv-flags needs the prose ids: the runner checks them for a server arm.
#   mtp-past     oursmtp:2044 at C 2048, N 4: refused before the lease by name (2044 + 4 + 1 = 2049), rc 64;
#                the dry run of 2044 and oursmtp:2043 beside it: rc 0, both arms' lines.
#   mtp-env      BLOOMERY_DRAFT=mtp in the runner's environment beside an ours arm: refused by name, rc 64.
#   mtp-dry      6 oursmtp:6's dry run: the oursmtp command line is ours' under env BLOOMERY_DRAFT=mtp, the
#                ours one carries no env, and the `[dry] oursmtp:` line says the checked-in schema declares
#                the mtp summary record (generate_glm5next registers mtp_summary).
#   mtp          6 oursmtp:6 lcpp27752srv:6, two rounds, preheat on at GLM_PREHEAT_K 2 on the fixture (needs
#                the prose ids): the stub generate_glm5next times its steps at half under BLOOMERY_DRAFT=mtp
#                and logs each run's BLOOMERY_DRAFT ($TMPDIR/stub-gen-draft), printing the mtp summary record
#                (STUB_GEN_MTP_REC=1): three runs of each arm, the oursmtp ones alone under mtp; oursmtp's
#                warm-up and rows, its K=2 preheats, its decode and prefill means under its own label, the
#                draft fields in its two rows and none in ours', its xcheck against ours;
#                `ratio mtp d=6 ours/oursmtp mean 0.4999 ± 0.0000 (n=2)`, and oursmtp in no other ratio
#                table (the decode and prefill tables hold ours/lcpp27752srv alone).
#   mtp-rec      the checked-in schema's mtp summary record: with the stub
#                printing one under mtp (STUB_GEN_MTP_REC=1) the oursmtp row carries `mtp positions/pass
#                2.500 = positions 10 / passes 4, kept [1, 1, 1, 1]` and the ours row nothing; printing
#                none, the oursmtp row is a FAIL row naming it, rc 1.
#   mtp-lever    a generate_glm5next that refuses BLOOMERY_DRAFT (STUB_GEN_NODRAFT=1): oursmtp:6 refused
#                before the lease by name with the binary's line, rc 2, no row.
#   lever-refuse each refusal of a lever arm's list (the runner's header's <D>@NAME=VALUE), one run an arm
#                beside a plain 512: rc 64 and the usage line ending in its reason, no [config], no lease, no
#                row, and no stub generate_glm5next run ($TMPDIR/stub-gen-env stays empty). Red on the runner
#                before the lever arms: its usage names no reason.
#   lever-dry    512 512@BLOOMERY_RESIDENCY=mid-p33-s1 oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1,BLOOMERY_R8=off's
#                dry run: the lever arms' command lines under env with their variables (oursmtp's after
#                BLOOMERY_DRAFT=mtp), the plain one with none, and the `[dry] residency:` line saying the
#                checked-in schema declares the residency records (residency_lever and residency_pass).
#   lever-pair   512 512@BLOOMERY_RESIDENCY=mid-p33-s1, two rounds, no warm-up, on the checked-in schema
#                (it declares the residency lever and pass records): four processes, the
#                stub engine seeing the variable in the lever arm's alone, in the rotation's order (none,
#                mid, mid, none); the lever rows' residency column (three step passes, kept 30), the plain
#                rows' none; two means with distinct labels, the ratio line ours/ours@… 1.0000 over two
#                rounds, the residency mean line of the lever label alone, no xcheck line for it; rc 0.
#                Needs the prose ids, as ours-timed; so do the three below.
#   lever-res-nolever  the same arms on the checked-in schema, the stub printing no residency lever record
#                (STUB_GEN_NORESLEVER=1): the lever row a FAIL row naming it, the plain row a ROW, rc 1.
#   lever-res-none  the same arms on a copy of the checked-in schema without its residency lever record (the
#                schema before generate_glm5next registered it): each lever row a FAIL row naming the schema,
#                the plain row a ROW, rc 1.
#   mtp-lever-arm  512 oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1 on the checked-in schema, the stub printing
#                the mtp summary record (STUB_GEN_MTP_REC=1): the stub engine
#                sees both BLOOMERY_DRAFT=mtp and the variable in the oursmtp process; its row under its label,
#                with its residency column; no ratio line and no xcheck line for it; rc 0.
#   lever-levers a generate_glm5next whose --levers refuses BLOOMERY_R8 (STUB_GEN_REFUSE): 512@BLOOMERY_R8=off
#                refused before the lease by name with the binary's line, rc 2, no row.
# The aggregate arm (the runner's <D>@BLOOMERY_GEN_SLOTS=N; red on the runner before it, which reads the arm as a
# plain lever arm):
#   slots-env    BLOOMERY_GEN_SLOTS=2 in the runner's own environment: refused by name, rc 64, before any row.
#   slots-bad    6@BLOOMERY_GEN_SLOTS=02: no slot count, refused by name, rc 64, before any row.
#   slots        6@BLOOMERY_RESIDENCY=mid-p0-s1 6@BLOOMERY_GEN_SLOTS=2,BLOOMERY_RESIDENCY=mid-p0-s1, two rounds, no
#                warm-up: the lever and 12 fed ids in the aggregate arm's processes alone (6 and none in its
#                twin's), its rows `tok/s(aggregate) 45.45 @ n=2·3` (2 positions a round, 3 rounds at 44 ms) with
#                prompt ids 50000..50011, its mean line `(aggregate of 2 slots)`, one `ratio slots d=6 <it>/<its
#                twin> mean 1.5000` (the twin at 33 ms a step) and no `ratio d=` line for it; rc 0. Needs the
#                prose ids, as ours-timed; so do the cases below.
#   slots-res-off  6@BLOOMERY_GEN_SLOTS=2, the residency off (generate_glm5next's unset), one round: no residency
#                record, so the residency clause has no pass to read and its row `tok/s(aggregate) 45.45 @ n=2·3`
#                stands on its time pass records, with no residency column; rc 0.
#   slots-pos, slots-leak, slots-none, slots-res-step  one FAIL row each, rc 1: a round of 1 position
#                (STUB_GEN_SLOTS_POS=1), kind=slots records from the plain arm 6 (STUB_GEN_SLOTS_LEAK=2), the
#                aggregate arm printing the plain lines (STUB_GEN_SLOTS_NOPASS=1), and a step pass in place of the
#                first pass=slots (STUB_GEN_SLOTS_RESPASS=step, rc=residency).
#   slots-past   the 2-slot arm of a D whose plain arm the prose file holds and whose 2·D ids it does not:
#                refused before the lease by name, rc 64.
#   slots-dry    the dry run of the twin and the aggregate arm: the aggregate arm's `slots=2 feed=12 twin=…`
#                and its command line under env with both variables and the ids 50000..50011, the twin's
#                with 50000..50005.
# An ours arm's place= and the two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090; depth-stub-cards.sh's
# nvidia-smi and journalctl first on PATH; red on the runner before them, which has no two-card mode, refuses
# place= as no lever row and BLOOMERY_GEN_PLACE=bp as no word):
#   bp-onecard, bp-onecard-arm  BLOOMERY_GEN_PLACE=bp and 6@place=bp under one card: refused by name, rc 64.
#   place-ref, place-word, place-twice, place-card  place= on lcpp27752:6, place=b2, place given twice,
#                6@place=gate with the A6000 the timing card: each refused by name, rc 64.
#   twocard-gate 6@place=gate in the two-card mode: refused by name, rc 64.
#   twocard-place  BLOOMERY_GEN_PLACE=bp, 6 6@place=a lcpp27752:6, one round, the stub naming its load's cards
#                (STUB_GEN_CARDS_ON): the two ours rows at `place bp` and `place a`, every row `A6000+3090`,
#                the reference bench run with the A6000 alone as CUDA_VISIBLE_DEVICES and its row's device
#                that card, the ratio ours/ours@place=a, the [config] placements and two-card lines, the
#                witness's 3090 cap and Xid lines in each of its eight blocks, the `ratio place d=6
#                ours/ours@place=a` line; rc 0. Needs the prose ids, as ours-timed; so do the five below.
#   twocard-place-mtp  6 6@place=a oursmtp:6 oursmtp:6@place=a, two rounds (STUB_GEN_MTP_REC): the oursmtp
#                rows at bp and a, their `ratio place d=6 oursmtp/oursmtp@place=a` and `ratio place pp p=6`
#                lines beside ours' pair, four lines in that table; rc 0 (red on the runner before the
#                placement pairs, which prints no table for an oursmtp arm with an @ list).
#   twocard-place-order  oursmtp:512@place=a,BLOOMERY_RESIDENCY=mid-p33-s1 and
#                oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1,place=bp, two rounds, bp's steps at 15 ms
#                (STUB_GEN_SMS_BP) against a's 16.5: `ratio place d=512 <bp arm>/<a arm> mean 1.1000` by
#                round with the arms in either order; rc 0 (red on a pairing by the arms' order, which reads
#                a / bp = 0.9091 when the a arm comes first).
#   twocard-place-3090  an a arm whose load record names the 3090 (STUB_GEN_CARDS_A): its FAIL row; rc 1.
#   twocard-nocards  a load record with no cards field, as generate_glm5next's is today: the FAIL row
#                naming it; rc 1.
#   twocard-place-dry  the dry run: each ours arm's --place, the reference's CUDA_VISIBLE_DEVICES, the
#                precheck's ok, the placements line.
#   dry-same     DEPTH_GLM5NEXT_BASE set: the dry run of every arm kind the base runner knows, under the
#                base and under the runner tested, byte for byte less the lines this runner changed on
#                purpose: `[dry] preheat:`, `[dry] prompt:`, `[dry] WARMUP r0:`, `[dry] server arms:`, the
#                server arms' rows and command lines, and the prompt id range of the command lines
#                (skipped, and said so, without it).
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
  "$ROOT/tools/ref/lcpp-fit.sh" "$ROOT/tools/ref/lcpp-warm.sh" "$ROOT/tools/ref/cold-blocks.sh" \
  "$ROOT/tools/ref/gguf-ranges.py" "$ROOT/tools/ref/lever-arms.sh" "$ROOT/tools/ref/arm-place.sh" "$ROOT/tools/ref/slots-arm.sh" \
  "$T/tools/ref/"
cp "$ROOT/tools/ref/models/glm5next.sh" "$T/tools/ref/models/"
cp "$ROOT/tools/bloomery/records.py" "$T/tools/bloomery/"
# The lever registry, which a lever arm's NAME=VALUE list is checked against (before the stub binaries are
# written, so they are newer than every crates/ source).
mkdir -p "$T/crates/levers/src"
cp "$ROOT/crates/levers/src/registry.rs" "$T/crates/levers/src/"
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
# The fault counter: $STUB_MAJFLT, a number the stub engines add to. The cold tag's readers are
# machine-wide /proc/vmstat counters whose premise is the lease ("the lease keeps it to the arm's
# process", cold-blocks.sh); this test takes no lease, and the box's ambient page-fault churn would
# tag random stub arms [cold] on their ~0.1 s windows — the copies read the file instead, so a row is
# cold exactly where a case says. The stub generate adds to it under STUB_GEN_FAULT, after its fed line.
# shellcheck disable=SC2016 # the copies expand them when they run
{
  echo 'majflt_now() { cat "$STUB_MAJFLT"; }'
  echo 'majflt_mark() { awk -v f="$1" -v re="$2" -v src="$STUB_MAJFLT" '"'"'!s && re != "" && $0 ~ re { getline l < src; close(src); print l > f; close(f); s = 1 } { print; fflush() }'"'"'; }'
} >> "$T/tools/ref/cold-blocks.sh"
echo 'lcpp_srv_majflt() { cat "$STUB_MAJFLT"; }' >> "$T/tools/ref/lcpp-warm.sh"
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
# --n-cpu-moe STUB_BENCH_FAIL_K. It removes the file STUB_BENCH_RM names. It appends its CUDA_VISIBLE_DEVICES
# to $TMPDIR/stub-bench-cvd, and under STUB_BENCH_DEVLINES=1 prints ggml_cuda_init's device lines for the
# cards that variable names (two: the A6000 and the 3090; one: the A6000). --help lists --fit-target unless its tree is in STUB_BENCH_NO_FIT. Under
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
echo "${CUDA_VISIBLE_DEVICES:-}" >> "${TMPDIR:-/tmp}/stub-bench-cvd"
if [ -n "${STUB_BENCH_DEVLINES:-}" ]; then
  case ${CUDA_VISIBLE_DEVICES:-} in
    *,*)
      echo "ggml_cuda_init: found 2 CUDA devices (Total VRAM: 72663 MiB):"
      echo "  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB"
      echo "  Device 1: NVIDIA GeForce RTX 3090 (stub), compute capability 8.6, VMM: yes, VRAM: 24124 MiB"
      ;;
    *)
      echo "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 48539 MiB):"
      echo "  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB"
      ;;
  esac >&2
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
# The stub llama-server: tools/ref/stub-llama-server.py (its docstring has what it answers).
cp "$ROOT/tools/ref/stub-llama-server.py" "$T/pr27752/llama-server"
cp "$T/pr27752/llama-server" "$T/pr27754/llama-server"
# The stub generate_glm5next: a variable named `place` in its environment (an arm's place= item passed as a
# variable) ends it at rc 9. Its load record names no cards, as generate_glm5next's does not; under
# STUB_GEN_CARDS_ON=1 it names the cards its --place loads (the A6000 under a, STUB_GEN_CARDS_A in their
# stead; the 3090 under gate; both under bp). --records-schema prints the checked-in schema; --levers prints a line, or
# under STUB_GEN_NODRAFT=1 with BLOOMERY_DRAFT set refuses it (exit 1), as a run does then, and so for the
# variable STUB_GEN_REFUSE names when it is set; a run appends
# `draft=<BLOOMERY_DRAFT or none> depth=<ids>` to $TMPDIR/stub-gen-draft and `draft=<…> residency=<BLOOMERY_RESIDENCY
# or none> depth=<ids>` to $TMPDIR/stub-gen-env; under BLOOMERY_RESIDENCY it prints a `residency lever`
# record (why=set) before its plan (none under STUB_GEN_NORESLEVER=1), a `residency pass` of pass none before its feed and one of pass step
# (kept 10, landed 1, late 0, made 1, bytes 100) after each timed step; it times its steps at half under
# BLOOMERY_DRAFT=mtp (at STUB_GEN_SMS_BP ms under --place bp when that is set), and prints there an `mtp
# summary` record when STUB_GEN_MTP_REC=1; it prints the
# records a timed run prints, the `fed` line left out under
# STUB_GEN_NOFED=1; its load record names --ctx (STUB_GEN_LOAD_CTX in its stead), its tokens record token 0
# STUB_GEN_TOKEN0 (1000 by default) and then the step lines' ids. Under STUB_GEN_FAULT=<marker> it adds
# 100000 to the fault counter after its `fed` line and times its prompt and steps at 0.01 ms,
# so the row is [cold]; with STUB_GEN_FAULT_ONCE=1 only its first run does. Under BLOOMERY_RESIDENCY it prints
# a `residency host` record after its plan, as generate_glm5next does. Every run appends `slots=<BLOOMERY_GEN_SLOTS
# or none> depth=<ids>` to $TMPDIR/stub-gen-slots. Under BLOOMERY_GEN_SLOTS=N (N >= 2, or STUB_GEN_SLOTS_LEAK=N on
# any arm) it prints the several-slot arm's records: N windows of depth / N ids, each slot's fed, step 0, time
# prompt and tokens records in slot order, each round's step records and a `time pass … kind=slots` at 44 ms of
# N positions (STUB_GEN_SLOTS_POS in its stead), the SMOKE footer with positions and tok/s(positions), and under
# BLOOMERY_RESIDENCY the residency passes none/0, prompt/0 a slot, then pass=slots kept=N a round
# (STUB_GEN_SLOTS_RESPASS names another pass); STUB_GEN_SLOTS_NOPASS=1 prints the plain lines instead.
G=$T/target/release/generate_glm5next
# shellcheck disable=SC2016 # ${1:-} is the stub's own argument
printf '#!/usr/bin/env bash\n[ "${1:-}" != --records-schema ] || exec cat %q\n' "$T/tools/bloomery/schema/generate_glm5next.jsonl" > "$G"
cat >> "$G" << 'EOF'
if printenv place > /dev/null; then
  echo "error: a variable named place reached the binary (place=$(printenv place)): the runner passed an arm's place= item as a variable" >&2
  exit 9
fi
if [ -n "${STUB_GEN_NODRAFT:-}" ] && [ -n "${BLOOMERY_DRAFT+set}" ]; then
  echo "Error: BLOOMERY_DRAFT is set, and generate_glm5next does not act on it (stub)" >&2
  exit 1
fi
if [ -n "${STUB_GEN_REFUSE:-}" ] && printenv "$STUB_GEN_REFUSE" > /dev/null; then
  echo "Error: $STUB_GEN_REFUSE is set, and generate_glm5next does not act on it (stub)" >&2
  exit 1
fi
[ "${1:-}" != --levers ] || { echo "lever table (stub)"; exit 0; }
n=32 place=gate depth=0 ctx=2048
while [ $# -gt 0 ]; do
  case $1 in
    --tokens) depth=$(echo "$2" | tr ',' '\n' | grep -c .); shift ;; -n) n=$2; shift ;; --place) place=$2; shift ;;
    --ctx) ctx=$2; shift ;; --warm) shift ;;
  esac
  shift
done
echo "draft=${BLOOMERY_DRAFT:-none} depth=$depth" >> "${TMPDIR:-/tmp}/stub-gen-draft"
echo "draft=${BLOOMERY_DRAFT:-none} residency=${BLOOMERY_RESIDENCY:-none} depth=$depth" >> "${TMPDIR:-/tmp}/stub-gen-env"
rp() { [ -z "${BLOOMERY_RESIDENCY:-}" ] || echo "residency pass pass=$1 boundary=$2 kept=$3 landed=$4 late=0 made=$4 in_flight=0 bytes=$5 end_us=1 boundary_us=2 wait_us=0 issue_us=1 stage_us=0 prepare_us=0"; }
[ -z "${BLOOMERY_RESIDENCY:-}" ] || [ -n "${STUB_GEN_NORESLEVER:-}" ] || echo "residency lever residency=$BLOOMERY_RESIDENCY why=set"
echo "plan place=$place card=A6000 ctx_max=2048 card_experts=2627 (39941832704 B) host_experts=9469 (145536581632 B) host_shadow=0 B n_l=67..68 on 39 layers card_budget=none"
[ -z "${BLOOMERY_RESIDENCY:-}" ] || echo "residency host residency=$BLOOMERY_RESIDENCY pinned=0 churn_experts=8 churn_bytes=4096 headroom=65536 headroom_after=61440"
cards=''
if [ -n "${STUB_GEN_CARDS_ON:-}" ]; then
  case $place in
    a) cards=${STUB_GEN_CARDS_A:-[NVIDIA_RTX_A6000]} ;;
    gate) cards='[NVIDIA_GeForce_RTX_3090]' ;;
    *) cards='[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090]' ;;
  esac
fi
echo "load resident_bytes=0 ctx=${STUB_GEN_LOAD_CTX:-$ctx}${cards:+ cards=$cards} (stub)"
echo "capture graph_nodes=1348"
echo "slots=${BLOOMERY_GEN_SLOTS:-none} depth=$depth" >> "${TMPDIR:-/tmp}/stub-gen-slots"
ns=${STUB_GEN_SLOTS_LEAK:-${BLOOMERY_GEN_SLOTS:-1}}
if [ "$ns" -gt 1 ] && [ -z "${STUB_GEN_SLOTS_NOPASS:-}" ]; then
  # The several-slot arm's records in generate_glm5next's order: each slot's fed and step 0, then each slot's
  # time prompt, each round's step a slot and its time pass, each slot's tokens, the SMOKE footer with the
  # aggregate, the residency passes (the seed's, a prompt call a slot, a pass a round).
  w=$((depth / ns))
  for j in $(seq 0 $((ns - 1))); do
    echo "fed ids=$w first=[1, 2, 3, 4] last=[5, 6, 7, 8] depth_sequence_from=$w"
    echo "step 0 $((w - 1)) $((12 + j)) (the $w fed steps in 0.1 s, runtime value)"
  done
  for j in $(seq 0 $((ns - 1))); do echo "time prompt n=$w ms=100.0000 tok/s=$((w * 10)).00 passes=1 kind=batch"; done
  for i in $(seq 1 $((n - 1))); do
    for j in $(seq 0 $((ns - 1))); do echo "step $i $((w + i - 1)) $((1000 + 100 * j + i))"; done
    echo "time pass $i ms=44.0000 positions=${STUB_GEN_SLOTS_POS:-$ns} kind=slots"
  done
  for j in $(seq 0 $((ns - 1))); do
    echo "tokens [$((12 + j))$(for i in $(seq 1 $((n - 1))); do printf ', %s' $((1000 + 100 * j + i)); done)]"
  done
  echo "SMOKE mode=graph place=$place prompt_tokens=$w depth=$w generated=$n warm=0 steps=$((n - 1)) p50_ms=44.0000 mean_ms=44.0000 tok/s(p50)=22.73 positions=$(((n - 1) * ns)) tok/s(positions)=$(awk -v n="$ns" 'BEGIN { printf "%.2f", n * 1000 / 44 }')"
  rp none 0 0 0 0
  for j in $(seq 1 "$ns"); do rp prompt "$j" 0 0 0; done
  for i in $(seq 1 $((n - 1))); do rp "${STUB_GEN_SLOTS_RESPASS:-slots}" $((ns + i)) "$ns" 1 100; done
  exit 0
fi
rp none 0 0 0 0
[ -n "${STUB_GEN_NOFED:-}" ] || echo "fed ids=$depth first=[1, 2, 3, 4] last=[5, 6, 7, 8] depth_sequence_from=$depth"
pms=100.0000 sms=33.0000
[ "${BLOOMERY_DRAFT:-}" != mtp ] || sms=16.5000
[ -z "${STUB_GEN_SMS_BP:-}" ] || [ "$place" != bp ] || sms=$STUB_GEN_SMS_BP
if [ -n "${STUB_GEN_FAULT:-}" ] && { [ -z "${STUB_GEN_FAULT_ONCE:-}" ] || [ ! -e "$STUB_GEN_FAULT.done" ]; }; then
  touch "$STUB_GEN_FAULT.done"
  pms=0.0100 sms=0.0100
  # after the runner's mark reads the counter at the fed line
  sleep 0.5
  echo $(($(cat "$STUB_MAJFLT") + 100000)) > "$STUB_MAJFLT"
fi
echo "step 0 $((depth - 1)) 12 (the $depth fed steps in 0.1 s, runtime value)"
echo "time prompt n=$depth ms=$pms tok/s=$((depth * 10)).00 passes=$depth kind=steps"
for i in $(seq 1 $((n - 1))); do echo "step $i $((depth + i - 1)) $((1000 + i))"; done
for i in $(seq 1 $((n - 1))); do echo "time step $i ms=$sms"; rp step "$i" 10 1 100; done
echo "tokens [$(for i in $(seq 0 $((n - 1))); do [ "$i" = 0 ] && printf '%s' "${STUB_GEN_TOKEN0:-1000}" || printf ', %s' $((1000 + i)); done)]"
[ -z "${STUB_GEN_MTP_REC:-}" ] || [ "${BLOOMERY_DRAFT:-}" != mtp ] ||
  echo "mtp summary proposals=3 kept=[1, 1, 1, 1] positions=10 passes=4 tok/s(positions)=200.00"
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
  echo 0 > "$tmp/tmp/stub-majflt"
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" STUB_MAJFLT="$tmp/tmp/stub-majflt" \
    BLOOMERY_MODEL=glm5next BLOOMERY_REF_MODEL_PROFILE=glm5next \
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

L=$tmp/srv-past.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- lcpp27754srv:2046
if [ "$RC" != 64 ]; then
  fail srv-past "rc $RC, want 64" "$L"
elif want srv-past "$L" 1 "^depth-glm5next.sh: arm 'lcpp27754srv:2046' is .* — the server's -c is ours' --ctx 2048 \(docs/fair-measure.md 1.5\), and 2046 ids \+ n_predict 4 \+ 1 take 2051 positions; raise BLOOMERY_GEN_CTX$"; then
  pass srv-past
fi
L=$tmp/pair.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 BLOOMERY_GEN_PAIR=1 -- 6
L2=$tmp/pair-bad.log
RC1=$RC
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_GEN_PAIR=yes -- 6
if [ "$RC1" != 0 ] || [ "$RC" != 64 ]; then
  fail pair "rc $RC1 (1) and $RC (yes), want 0 and 64" "$L2"
elif want pair "$L" 1 '^\[dry\]     timeout --kill-after=10 60 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50005> -n 4 --ctx 2048 --place a --time --pair $' &&
  want pair "$L2" 1 "^depth-glm5next.sh: BLOOMERY_GEN_PAIR is 1 or unset, got 'yes'$" &&
  want pair "$L2" 0 '^(ROW|WARMUP|FAIL) '; then
  pass pair
fi
L=$tmp/srv-flags-bad.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- lcpp27754srv+x1:6
L2=$tmp/srv-flags-bench.log
RC1=$RC
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 -- lcpp27754+t16:6
if [ "$RC1" != 64 ] || [ "$RC" != 64 ]; then
  fail srv-flags-refused "refusals: rc $RC1 and $RC, want 64 and 64" "$L"
elif want srv-flags-refused "$L" 1 "^depth-glm5next.sh: arm 'lcpp27754srv\+x1:6' is .* — .*\+x1" &&
  want srv-flags-refused "$L2" 1 "^depth-glm5next.sh: arm 'lcpp27754\+t16:6' is .* — \+t<N>, \+nopo<0\|1> and \+k<K> are a server arm's flags$"; then
  pass srv-flags-refused
fi
L=$tmp/mtp-past.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- oursmtp:2044
L2=$tmp/mtp-past-dry.log
RC1=$RC
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- 2044 oursmtp:2043
if [ "$RC1" != 64 ] || [ "$RC" != 0 ]; then
  fail mtp-past "rc $RC1 (2044) and $RC (the dry run beside it), want 64 and 0" "$L"
elif want mtp-past "$L" 1 "^depth-glm5next.sh: arm 'oursmtp:2044' is .* — oursmtp needs 1 <= D and D \+ N \+ 1 <= C \(2044 \+ 4 \+ 1 = 2049 against --ctx 2048\): the verify's last window takes one position past the N-th token$" &&
  want mtp-past "$L" 0 '^(ROW|WARMUP|FAIL) ' &&
  want mtp-past "$L2" 1 '^\[dry\] 2044:$' &&
  want mtp-past "$L2" 1 '^\[dry\] oursmtp:2043:$' &&
  want mtp-past "$L2" 1 '^\[dry\]     timeout --kill-after=10 60 env BLOOMERY_DRAFT=mtp target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.52042> -n 4 --ctx 2048 '; then
  pass mtp-past
fi
L=$tmp/mtp-env.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRAFT=mtp -- 6 oursmtp:6
if [ "$RC" != 64 ]; then
  fail mtp-env "rc $RC, want 64" "$L"
elif want mtp-env "$L" 1 "^depth-glm5next.sh: BLOOMERY_DRAFT is set to 'mtp' in the runner's environment, so the plain ours arms would run it too; unset it \(the oursmtp:<D> arm sets BLOOMERY_DRAFT=mtp for its own command\)$" &&
  want mtp-env "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]) '; then
  pass mtp-env
fi
L=$tmp/mtp-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- 6 oursmtp:6
if [ "$RC" != 0 ]; then
  fail mtp-dry "rc $RC, want 0" "$L"
elif want mtp-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 env BLOOMERY_DRAFT=mtp target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50005> -n 4 --ctx 2048 --place a --time $' &&
  want mtp-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50005> -n 4 --ctx 2048 --place a --time $' &&
  want mtp-dry "$L" 2 'BLOOMERY_DRAFT' &&
  want mtp-dry "$L" 1 "^\[dry\] oursmtp: ours' command under env BLOOMERY_DRAFT=mtp; generate_glm5next's checked-in schema declares the mtp summary record: each oursmtp row carries its positions, passes and kept, and one that prints none is a FAIL row$" &&
  want mtp-dry "$L" 1 '^\[dry\] round 1 gguf order: 6 oursmtp:6 $'; then
  pass mtp-dry
fi
# The lever arms' refusals: each an arm beside a plain 512, refused by name before anything runs.
REG=tools/ref/../../crates/levers/src/registry.rs
# refuse_case <name> <arm> <reason> [NAME=VALUE...]: rc 64, the usage line of <arm> ending in ` — <reason>`,
# nothing past the parse.
refuse_case() {
  local name=$1 arm=$2 why=$3 log=$tmp/lever-refuse-$1.log
  shift 3
  : > "$tmp/tmp/stub-gen-env"
  stub_run "$log" BLOOMERY_AB_ROUNDS=1 "$@" -- 512 "$arm"
  if [ "$RC" != 64 ]; then
    fail "lever-refuse $name" "rc $RC, want 64" "$log"
  elif ! grep -F -- "depth-glm5next.sh: arm '$arm' is " "$log" | grep -qF -- " — $why"; then
    fail "lever-refuse $name" "no usage line of '$arm' ending in: $why" "$log"
  elif want "lever-refuse $name" "$log" 0 '^(ROW|WARMUP|FAIL|\[config\]|\[stub\] no lease)' &&
    want "lever-refuse $name" "$tmp/tmp/stub-gen-env" 0 '.'; then
    pass "lever-refuse $name"
  fi
}
refuse_case empty-list '512@' "an empty NAME=VALUE list after '@'"
refuse_case space '512@BLOOMERY_RESIDENCY=off BLOOMERY_R8=off' "white space in 'BLOOMERY_RESIDENCY=off BLOOMERY_R8=off' (a value holds none)"
refuse_case empty-item '512@BLOOMERY_RESIDENCY=off,,BLOOMERY_R8=off' "an empty item in 'BLOOMERY_RESIDENCY=off,,BLOOMERY_R8=off' (NAME=VALUE items are separated by one ',')"
refuse_case not-kv '512@BLOOMERY_RESIDENCY' "'BLOOMERY_RESIDENCY' is no NAME=VALUE (a ',' separates two variables, so a value holds none)"
refuse_case empty-name '512@=off' "'=off' has an empty name"
refuse_case empty-value '512@BLOOMERY_RESIDENCY=' "BLOOMERY_RESIDENCY has an empty value"
refuse_case bad-name '512@1BLOOMERY=off' "'1BLOOMERY' is no variable name"
refuse_case at-value '512@BLOOMERY_RESIDENCY=mid@p33' "the value of BLOOMERY_RESIDENCY holds '@', which opens an arm's variables"
refuse_case pipe-value '512@BLOOMERY_RESIDENCY=mid|p33' "the value of BLOOMERY_RESIDENCY holds '|', the separator of this runner's records"
refuse_case twice '512@BLOOMERY_RESIDENCY=off,BLOOMERY_RESIDENCY=mid-p33-s1' "BLOOMERY_RESIDENCY is given twice"
refuse_case ref-lcpp 'lcpp27754:512@BLOOMERY_RESIDENCY=off' "'@' sets a lever of ours, and lcpp27754: is a reference engine's arm — a lever of ours is not a reference's"
refuse_case ref-exl3 'exl3:256@BLOOMERY_RESIDENCY=off' "'@' sets a lever of ours, and exl3: is a reference engine's arm — a lever of ours is not a reference's"
refuse_case mtp-draft 'oursmtp:512@BLOOMERY_DRAFT=x' "BLOOMERY_DRAFT is the oursmtp arm's own (it sets BLOOMERY_DRAFT=mtp): give the arm its other variables only"
refuse_case ours-draft '512@BLOOMERY_DRAFT=mtp' "BLOOMERY_DRAFT is the oursmtp arm's own: the drafted arm is oursmtp:512[@…], which sets BLOOMERY_DRAFT=mtp"
# A name no registry row names, spelt in two parts so check-levers does not read it as one.
NOPE="BLOOMERY""_NOPE"
refuse_case no-row "512@$NOPE=1" "$NOPE is no row of the lever registry ($REG): a lever arm sets a lever"
refuse_case ab-load '512@BLOOMERY_AB_LOAD=arm' "BLOOMERY_AB_LOAD is no lever: its row in $REG is a runner's, a harness's or a path's own variable, not a setting of the binary's"
refuse_case runner-env '512@BLOOMERY_RESIDENCY=off' "BLOOMERY_RESIDENCY is set in the runner's own environment (BLOOMERY_RESIDENCY=off), which every arm inherits, so the plain arms' labels would not show it: give it per arm only" BLOOMERY_RESIDENCY=off

L=$tmp/lever-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- 512 512@BLOOMERY_RESIDENCY=mid-p33-s1 oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1,BLOOMERY_R8=off
if [ "$RC" != 0 ]; then
  fail lever-dry "rc $RC, want 0" "$L"
elif want lever-dry "$L" 1 '^\[dry\] 512@BLOOMERY_RESIDENCY=mid-p33-s1:$' &&
  want lever-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 env BLOOMERY_RESIDENCY=mid-p33-s1 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50511> -n 4 --ctx 2048 --place a --time $' &&
  want lever-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 env BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=mid-p33-s1 BLOOMERY_R8=off target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50511> -n 4 --ctx 2048 --place a --time $' &&
  want lever-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50511> -n 4 --ctx 2048 --place a --time $' &&
  want lever-dry "$L" 1 "^\\[dry\\] residency: generate_glm5next's checked-in schema declares the residency records: each ours row carries its residency lever and passes$" &&
  want lever-dry "$L" 1 '^\[dry\] round 1 gguf order: 512 512@BLOOMERY_RESIDENCY=mid-p33-s1 oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1,BLOOMERY_R8=off $'; then
  pass lever-dry
fi
# The two-card mode's stubs, apart from this test's one-card nvidia-smi: depth-stub-cards.sh's nvidia-smi
# (the cards by UUID, their power limits and buses) and journalctl (the Xid lines) under $TCB/bin, first on
# PATH in the two-card cases only.
TCB=$tmp/tcbin
mkdir -p "$TCB/bin" "$TCB/tools/ref"
(T=$TCB && . "$HERE/depth-stub-cards.sh")
chmod +x "$TCB/bin/"*
TC=(BLOOMERY_TIMING_CARDS=a6000+3090 PATH="$TCB/bin:$T/bin:$PATH")
# place_refused <name> <pattern> <env…> -- <arms…>: the run refused before anything runs, rc 64, the pattern
# on one line, no row, witness or lease. Red on the runner before a per-arm placement: it refuses `place` as
# no lever row, BLOOMERY_GEN_PLACE=bp as no word, and runs `gate` in the mode on one card.
place_refused() {
  local name=$1 pat=$2
  shift 2
  L=$tmp/$name.log
  stub_run "$L" "$@"
  if [ "$RC" != 64 ]; then
    fail "$name" "rc $RC, want 64" "$L"
  elif want "$name" "$L" 1 "$pat" && want "$name" "$L" 0 '^(ROW|FAIL|DISCARD|WARMUP) |^--- witness|^\[stub\] no lease'; then
    pass "$name"
  fi
}
place_refused bp-onecard "^depth-glm5next.sh: BLOOMERY_GEN_PLACE=bp is plan \(b′\), which loads on both cards \(the A6000 and its 3090 expert tier\); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000\+3090$" BLOOMERY_GEN_PLACE=bp -- 6
place_refused bp-onecard-arm "^depth-glm5next.sh: arm '6@place=bp': place=bp is plan \(b′\), which loads on both cards \(the A6000 and its 3090 expert tier\); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000\+3090$" -- 6 6@place=bp
place_refused place-ref "^depth-glm5next.sh: arm 'lcpp27752:6@place=a': place= sets generate_glm5next's --place, and lcpp27752 is a reference engine's arm, which takes no placement of ours$" -- 6 lcpp27752:6@place=a
place_refused place-word "^depth-glm5next.sh: arm '6@place=b2': place=b2: generate_glm5next takes --place a, gate or bp in this runner$" -- 6 6@place=b2
place_refused place-twice "^depth-glm5next.sh: arm '6@place=a,place=a': place is given twice$" -- 6 6@place=a,place=a
place_refused place-card "^depth-glm5next.sh: arm '6@place=gate': place=gate is the gate plan, which loads on the 3090, and the timing card is " -- 6 6@place=gate
place_refused twocard-gate "^depth-glm5next.sh: arm '6@place=gate': place=gate is the gate plan, which loads the 3090 alone; the two-card mode times a \(plan \(a\) on the A6000, the 3090 idle\) or bp \(plan \(b′\), both cards\)$" "${TC[@]}" -- 6 6@place=gate

# The aggregate arm's refusals before anything runs (slots-arm.sh's slots_env_check and arm_slots): the lever in
# the runner's own environment, and an N that is no slot count.
L=$tmp/slots-env.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_GEN_SLOTS=2 -- 6
if [ "$RC" != 64 ]; then
  fail slots-env "rc $RC, want 64" "$L"
elif want slots-env "$L" 1 "^depth-glm5next.sh: BLOOMERY_GEN_SLOTS=2 is set in the runner's own environment, which every arm inherits: " &&
  want slots-env "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]) '; then
  pass slots-env
fi
L=$tmp/slots-bad.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- 6 6@BLOOMERY_GEN_SLOTS=02
if [ "$RC" != 64 ]; then
  fail slots-bad "rc $RC, want 64" "$L"
elif want slots-bad "$L" 1 "^depth-glm5next.sh: arm '6@BLOOMERY_GEN_SLOTS=02': BLOOMERY_GEN_SLOTS=02 is no slot count \(a whole number from 1, no leading zero\)" &&
  want slots-bad "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]) '; then
  pass slots-bad
fi

PROSE_SRC=${BLOOMERY_DATA:-/root/bloomery-data}/glm5next/corpus-prose.ids
if [ -f "$PROSE_SRC" ]; then
  mkdir -p "$T/data/glm5next"
  cp "$PROSE_SRC" "$T/data/glm5next/"
  GPU_A=$STUB_GPU_A6000
  L=$tmp/ours-timed.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" "${PH_ENV[@]}" -- 6
  L2=$tmp/ours-nofed.log
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_NOFED=1 GLM_PROSE_FROM=0 -- 6
  if [ "$RC1" != 0 ]; then
    fail ours-timed "rc $RC1, want 0" "$L"
  elif [ "$RC" != 0 ]; then
    fail ours-timed "no fed line: rc $RC, want 0" "$L2"
  elif want ours-timed "$L" 1 '^ROW r1 ours d=6 n=4 ctx=2048 \| .*\| prompt ids 50000..50005 \| .*\| majflt [0-9]+ \(timed, from the fed line; whole process [0-9]+\) <= [0-9.]+ % of the timed window \| wall [0-9]+s$' &&
    want ours-timed "$L" 1 '^WARMUP r0 ours d=6 .*\| majflt [0-9]+ \(timed, from the fed line; ' &&
    want ours-timed "$L" 2 '^preheat ours K=2 bytes=2136 ' &&
    want ours-timed "$L" 1 '^\[config\] prompt: GLM_PROSE ids from index 50000$' &&
    want ours-timed "$L" 0 '\(whole process\)' &&
    want ours-timed "$L2" 1 '^ROW r1 ours d=6 .*\| majflt [0-9]+ \(whole process: no fed line\) <= ' &&
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

  # The server arms (lcpp-warm.sh's): flags, context, the warm-up request, draft, prefill rows, cross-check.
  L=$tmp/srv.log
  : > "$tmp/tmp/stub-srv-argv"
  : > "$tmp/tmp/stub-srv-reqs"
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" -- 6 lcpp27754srv:6 lcpp27752mtp:6 lcpp27754srvpp:6
  ids6=$(sed -n '50001,50006p' "$T/data/glm5next/corpus-prose.ids" | paste -sd, -)
  if [ "$RC" != 0 ]; then
    fail srv "rc $RC, want 0" "$L"
  elif want srv "$L" 1 '^ROW r1 lcpp27754srv d=6 n=4 \| tok/s 20\.00 @ n=4, depth 6, A6000 \(stub\) \| llama-server /completion \| prompt ids 50000\.\.50005 \| warm-up 20\.00 tok/s majflt [0-9]+, continuation [^|]*\| -c 2048 \| prompt_n 6 prompt tok/s 60\.00 \| draft_n 0 draft_n_accepted 0 \| no draft acceptance line \| majflt ' &&
    want srv "$L" 1 '^ROW r1 lcpp27752mtp d=6 n=4 \| .*\| draft_n 4 draft_n_accepted 3 \| draft acceptance = 0\.75000 \( +3 accepted / +4 generated\), mean len = 1\.75 \| ' &&
    want srv "$L" 1 '^ROW r1 lcpp27754srvpp p=6 n=0 \| tok/s\(pp\) 60\.00 @ n=0, prompt 6, A6000 \(stub\) \| llama-server /completion \| prompt ids 50000\.\.50005 \| warm-up 60\.00 tok/s\(pp\) majflt [0-9]+, continuation [^|]*\| -c 2048 \| ub 512 b 2048 \(llama-server defaults\) \| ' &&
    want srv "$L" 1 '^WARMUP r0 ours d=6 ' &&
    want srv "$L" 0 '^WARMUP r0 lcpp2775' &&
    want srv "$L" 1 '^mean lcpp27754srv +6 +60\.00 tok/s\(pp\) ' &&
    want srv "$L" 1 '^mean lcpp27754srv +6 +20\.00 tok/s ' &&
    want srv "$L" 1 '^xcheck prose p=6 r1 ours\(r1\)/lcpp27754srv: same 4 \(1000,1001,1002,1003\)$' &&
    want srv "$L" 1 '^xcheck prose p=6 r1 ours\(r1\)/lcpp27752mtp: same 4 ' &&
    want srv "$L" 1 '^xcheck prose p=6 r1 ours\(r1\)/lcpp27754srvpp: same 1 \(1000\)$' &&
    want srv "$L" 0 '^failed arms' &&
    want "srv reqs" "$tmp/tmp/stub-srv-reqs" 4 "^[12] 6 4 $ids6$" &&
    want "srv reqs" "$tmp/tmp/stub-srv-reqs" 2 "^[12] 6 1 $ids6$" &&
    want "srv argv" "$tmp/tmp/stub-srv-argv" 2 ' -ngl 999 --n-cpu-moe 36 -fa off -t 32 --no-op-offload -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 2048 --host 127\.0\.0\.1 --port 0$' &&
    want "srv argv" "$tmp/tmp/stub-srv-argv" 1 ' -ngl 999 -fa on -t 32 --no-op-offload --n-cpu-moe 37 -fit off --spec-type draft-mtp --spec-draft-n-max 2 -np 1 -ctxcp 0 --cache-ram 0 -c 2048 ' &&
    { [ -z "$(while read -r p; do kill -0 "$p" 2> /dev/null && echo "$p"; done < "$tmp/tmp/stub-srv-pids")" ] ||
      { fail srv "a stub server is still up after the run" "$L"; false; }; }; then
    pass srv
  fi
  L=$tmp/srv-xcheck.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_TOKEN0=7 -- 6 lcpp27754srv:6
  if [ "$RC" != 1 ]; then
    fail srv-xcheck "rc $RC, want 1" "$L"
  elif want srv-xcheck "$L" 1 '^FAIL xcheck prose p=6 r1 ours\(r1\)/lcpp27754srv: differs at 0 of 4 \| ours 7,1001,1002,1003 \| lcpp27754srv 1000,1001,1002,1003$' &&
    want srv-xcheck "$L" 1 '^failed arms: r1 xcheck lcpp27754srv p=6 \(differs at 0 of 4\)$' &&
    want srv-xcheck "$L" 1 '^    dropped: lcpp27754srv at 6 \(FAIL xcheck\)$' &&
    want srv-xcheck "$L" 0 '^mean lcpp27754srv '; then
    L=$tmp/srv-xcheck-tail.log
    stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_SRV_TOKEN1=7 -- 6 lcpp27754srv:6
    if [ "$RC" != 0 ]; then
      fail srv-xcheck "tail: rc $RC, want 0" "$L"
    elif want srv-xcheck "$L" 1 '^xcheck prose p=6 r1 ours\(r1\)/lcpp27754srv: first id same, differs at 1 of 4 \| ours 1000,1001,1002,1003 \| lcpp27754srv 1000,7,1002,1003 \[xcheck-tail\]$' &&
      want srv-xcheck "$L" 1 '^xcheck: 1 server row\(s\): 0 same, 1 \[xcheck-tail\] ' &&
      want srv-xcheck "$L" 1 '^mean lcpp27754srv +6 +20\.00 tok/s '; then
      pass srv-xcheck
    fi
  fi
  L=$tmp/srv-ctx.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_SRV_NCTX=4096 -- lcpp27754srv:6
  L2=$tmp/srv-loadctx.log
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_LOAD_CTX=4096 -- 6
  if [ "$RC1" != 1 ] || [ "$RC" != 1 ]; then
    fail srv-ctx "rc $RC1 (server) and $RC (ours), want 1 and 1" "$L"
  elif want srv-ctx "$L" 1 '^FAIL r1 lcpp27754srv d=6 rc=0 \| the server made its context at n_ctx 4096, not the -c 2048 ours runs at \(docs/fair-measure.md 1.5\) \| ' &&
    want srv-ctx "$L2" 1 "^FAIL r1 ours d=6 rc=0 \| its load record names ctx=4096, and the server arms run at -c 2048 \(BLOOMERY_GEN_CTX, ours' --ctx\) \| "; then
    pass srv-ctx
  fi
  L=$tmp/srv-flags.log
  : > "$tmp/tmp/stub-srv-argv"
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- lcpp27754srv+t16+k40:6
  if [ "$RC" != 0 ]; then
    fail srv-flags "rc $RC, want 0" "$L"
  elif want srv-flags "$L" 1 '^ROW r1 lcpp27754srv\+t16\+k40 d=6 n=4 \| ' &&
    want "srv-flags argv" "$tmp/tmp/stub-srv-argv" 1 ' -ngl 999 -fa off --no-op-offload -t 16 --n-cpu-moe 40 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 2048 '; then
    pass srv-flags
  fi
  L=$tmp/srv-dry.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 -- lcpp27754srv:6 lcpp27752mtp:6
  if [ "$RC" != 0 ]; then
    fail srv-dry "rc $RC, want 0" "$L"
  elif want srv-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env NVIDIA_TF32_OVERRIDE=0 [^ ]*/pr27754/llama-server -m [^ ]* -ngl 999 --n-cpu-moe 36 -fa off -t 32 --no-op-offload -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 2048 --host 127\.0\.0\.1 --port 0 $" &&
    want srv-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 [^ ]*/pr27752/llama-server -m [^ ]* -ngl 999 -fa on -t 32 --no-op-offload --n-cpu-moe 37 -fit off --spec-type draft-mtp --spec-draft-n-max 2 -np 1 -ctxcp 0 --cache-ram 0 -c 2048 --host 127\.0\.0\.1 --port 0 $" &&
    want srv-dry "$L" 1 "^\[dry\] server arms: -c 2048 \(ours' --ctx\), a discarded request then the timed one; " &&
    want srv-dry "$L" 0 '^\[dry\] check: '; then
    pass srv-dry
  fi
  L=$tmp/srv-probe.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_SRV_HELP_MISSING=--spec-type -- lcpp27752mtp:6
  if [ "$RC" != 64 ]; then
    fail srv-probe "rc $RC, want 64" "$L"
  elif want srv-probe "$L" 1 "^depth-glm5next.sh: arm lcpp27752mtp:6: [^ ]*/pr27752/llama-server --help lists no --spec-type$"; then
    pass srv-probe
  fi
  L=$tmp/srv-fail.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_SRV_NODRAFT=1 -- lcpp27752mtp:6
  L2=$tmp/srv-exit.log
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_SRV_EXIT=1 -- lcpp27754srv:6
  if [ "$RC1" != 1 ] || [ "$RC" != 1 ]; then
    fail srv-fail "rc $RC1 (no draft) and $RC (exit), want 1 and 1" "$L"
  elif want srv-fail "$L" 1 '^FAIL r1 lcpp27752mtp d=6 rc=0 \| the MTP arm drafted nothing \(draft_n 0\) \| ' &&
    want srv-fail "$L2" 1 "^FAIL r1 lcpp27754srv d=6 rc=5 \| -c 2048 \(ours' --ctx\): llama-server exited 5 before it answered /health \| "; then
    pass srv-fail
  fi

  # The MTP arm: ours under BLOOMERY_DRAFT=mtp, its rows, preheat, xcheck and the ours / oursmtp table.
  L=$tmp/mtp.log
  : > "$tmp/tmp/stub-gen-draft"
  stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_MTP_REC=1 "${PH_ENV[@]}" -- 6 oursmtp:6 lcpp27752srv:6
  if [ "$RC" != 0 ]; then
    fail mtp "rc $RC, want 0" "$L"
  elif want mtp "$L" 1 '^WARMUP r0 oursmtp d=6 n=4 ctx=2048 \| ' &&
    want mtp "$L" 1 '^ROW r1 oursmtp d=6 n=4 ctx=2048 \| tok/s\(mean\) 60\.61 @ n=4, depth 6, A6000 \(stub\) \| .*\| prompt ids 50000\.\.50005 \| .*\| pp_tok/s 60\.00 \(n=6, passes=6, kind=steps\) \| mtp positions/pass 2\.500 = positions 10 / passes 4, kept \[1, 1, 1, 1\] \| majflt [0-9]+ \(timed, from the fed line; ' &&
    want mtp "$L" 1 '^ROW r2 oursmtp d=6 ' &&
    want mtp "$L" 1 '^ROW r1 ours d=6 n=4 ctx=2048 \| tok/s\(mean\) 30\.30 @ ' &&
    want mtp "$L" 2 '^ROW r[12] oursmtp d=6 .*\| mtp positions/pass ' &&
    want mtp "$L" 0 '^ROW r[12] ours d=6 .*\| mtp positions/pass ' &&
    want mtp "$L" 3 '^preheat oursmtp K=2 bytes=2136 ' &&
    want mtp "$L" 3 '^preheat ours K=2 bytes=2136 ' &&
    want mtp "$L" 1 "^\[config\] oursmtp: ours' command under env BLOOMERY_DRAFT=mtp; generate_glm5next's checked-in schema declares the mtp summary record: each oursmtp row carries its positions, passes and kept, and one that prints none is a FAIL row$" &&
    want mtp "$L" 1 '^mean oursmtp 6 +60\.61 tok/s ' &&
    want mtp "$L" 1 '^mean ours 6 +30\.30 tok/s ' &&
    want mtp "$L" 1 '^mean oursmtp 6 +60\.00 tok/s\(pp\) ' &&
    want mtp "$L" 1 '^xcheck prose p=6 r1 ours\(r1\)/oursmtp: same 4 \(1000,1001,1002,1003\)$' &&
    want mtp "$L" 1 '^xcheck prose p=6 r2 ours\(r2\)/oursmtp: same 4 ' &&
    want mtp "$L" 1 '^ratio mtp d=6 +ours/oursmtp +mean 0\.4999 ± 0\.0000 \(n=2\)  of means 0\.4999  per round: r1 0\.4999 r2 0\.4999$' &&
    want mtp "$L" 1 '^ratio mtp d=' &&
    want mtp "$L" 1 '^ratio d=6 +ours/lcpp27752srv ' &&
    want mtp "$L" 1 '^ratio d=' &&
    want mtp "$L" 1 '^ratio pp p=6 +ours/lcpp27752srv ' &&
    want mtp "$L" 1 '^ratio pp p=' &&
    want mtp "$L" 0 '^failed arms' &&
    want "mtp draft" "$tmp/tmp/stub-gen-draft" 3 '^draft=mtp depth=6$' &&
    want "mtp draft" "$tmp/tmp/stub-gen-draft" 3 '^draft=none depth=6$' &&
    want "mtp draft" "$tmp/tmp/stub-gen-draft" 6 '.'; then
    pass mtp
  fi

  # The mtp summary record the checked-in schema declares: printed, its fields; not printed, a FAIL row.
  SCH=$T/tools/bloomery/schema/generate_glm5next.jsonl
  L=$tmp/mtp-rec.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_MTP_REC=1 -- 6 oursmtp:6
  L2=$tmp/mtp-rec-none.log
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- 6 oursmtp:6
  if [ "$RC1" != 0 ] || [ "$RC" != 1 ]; then
    fail mtp-rec "rc $RC1 (a record) and $RC (none), want 0 and 1" "$L"
  elif want mtp-rec "$L" 1 '^ROW r1 oursmtp d=6 .*\| pp_tok/s 60\.00 \(n=6, passes=6, kind=steps\) \| mtp positions/pass 2\.500 = positions 10 / passes 4, kept \[1, 1, 1, 1\] \| majflt ' &&
    want mtp-rec "$L" 1 '^ROW r1 ours d=6 ' &&
    want mtp-rec "$L" 1 '\| mtp positions/pass ' &&
    want mtp-rec "$L" 1 "^\[config\] oursmtp: ours' command under env BLOOMERY_DRAFT=mtp; generate_glm5next's checked-in schema declares the mtp summary record: " &&
    want mtp-rec "$L2" 1 '^FAIL r1 oursmtp d=6 rc=0 \| the oursmtp arm printed no mtp summary record, which its schema declares \| full output: ' &&
    want mtp-rec "$L2" 1 '^ROW r1 ours d=6 ' &&
    want mtp-rec "$L2" 1 '^failed arms: r1:oursmtp@d=6$'; then
    pass mtp-rec
  fi

  L=$tmp/mtp-lever.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_NODRAFT=1 -- 6 oursmtp:6
  if [ "$RC" != 2 ]; then
    fail mtp-lever "rc $RC, want 2" "$L"
  elif want mtp-lever "$L" 1 '^depth-glm5next.sh: target/release/generate_glm5next refuses BLOOMERY_DRAFT=mtp \(its --levers under it: Error: BLOOMERY_DRAFT is set, and generate_glm5next does not act on it \(stub\)\): no oursmtp arm can run$' &&
    want mtp-lever "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]) '; then
    pass mtp-lever
  fi

  # The lever arms on the checked-in schema, which declares the residency lever and pass records; for
  # lever-res-none a copy without the residency lever record, the schema as it was before generate_glm5next
  # registered it.
  L=$tmp/lever-pair.log
  : > "$tmp/tmp/stub-gen-env"
  stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- 512 512@BLOOMERY_RESIDENCY=mid-p33-s1
  cp "$tmp/tmp/stub-gen-env" "$tmp/lever-pair-env.txt"
  L2=$tmp/mtp-lever-arm.log
  RC1=$RC
  : > "$tmp/tmp/stub-gen-env"
  stub_run "$L2" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_MTP_REC=1 -- 512 oursmtp:512@BLOOMERY_RESIDENCY=mid-p33-s1
  cp "$tmp/tmp/stub-gen-env" "$tmp/mtp-lever-arm-env.txt"
  RC2=$RC
  L3=$tmp/lever-res-nolever.log
  stub_run "$L3" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_NORESLEVER=1 -- 512 512@BLOOMERY_RESIDENCY=mid-p33-s1
  RC3=$RC
  LV=ours@BLOOMERY_RESIDENCY=mid-p33-s1
  if [ "$RC1" != 0 ]; then
    fail lever-pair "rc $RC1, want 0" "$L"
  elif [ "$(paste -sd' ' "$tmp/lever-pair-env.txt")" != "draft=none residency=none depth=512 draft=none residency=mid-p33-s1 depth=512 draft=none residency=mid-p33-s1 depth=512 draft=none residency=none depth=512" ]; then
    fail lever-pair "the stub engine's environment per process, want none, mid, mid, none (two loads a round)" "$tmp/lever-pair-env.txt"
  elif want lever-pair "$L" 1 "^ROW r1 $LV d=512 n=4 ctx=2048 \\| tok/s\\(mean\\) 30\\.30 @ n=4, depth 512, A6000 \\(stub\\) \\| .*\\| residency mid-p33-s1 \\(set\\) passes 3 kept 30 landed 3 late 0 made 3 bytes 300 \\| majflt " &&
    want lever-pair "$L" 1 "^ROW r2 $LV d=512 .*\\| residency mid-p33-s1 \\(set\\) passes 3 " &&
    want lever-pair "$L" 2 '^ROW r[12] ours d=512 n=4 ' &&
    want lever-pair "$L" 0 '^ROW r[12] ours d=512 .*\| residency ' &&
    want lever-pair "$L" 1 '^mean ours 512 +30\.30 tok/s ' &&
    want lever-pair "$L" 1 "^mean $LV 512 +30\\.30 tok/s " &&
    want lever-pair "$L" 1 "^ratio d=512 +ours/$LV +mean 1\\.0000 ± 0\\.0000 \\(n=2\\)  of means 1\\.0000  per round: r1 1\\.0000 r2 1\\.0000$" &&
    want lever-pair "$L" 1 "^ratio pp p=512 +ours/$LV " &&
    want lever-pair "$L" 1 "^residency mean $LV d=512 mid-p33-s1: rows 2, passes/row 3\\.0, kept/pass 10\\.00, landed/pass 1\\.000, late/pass 0\\.000, made/pass 1\\.000$" &&
    want lever-pair "$L" 1 '^residency mean ' &&
    want lever-pair "$L" 0 '^(FAIL )?xcheck .*ours@' &&
    want lever-pair "$L" 1 "^\\[config\\] residency: generate_glm5next's checked-in schema declares the residency records: " &&
    want lever-pair "$L" 0 '^failed arms'; then
    pass lever-pair
  fi
  if [ "$RC2" != 0 ]; then
    fail mtp-lever-arm "rc $RC2, want 0" "$L2"
  elif [ "$(paste -sd' ' "$tmp/mtp-lever-arm-env.txt")" != "draft=none residency=none depth=512 draft=mtp residency=mid-p33-s1 depth=512" ]; then
    fail mtp-lever-arm "the stub engine's environment per process, want the plain arm's none and the oursmtp arm's both" "$tmp/mtp-lever-arm-env.txt"
  elif want mtp-lever-arm "$L2" 1 '^ROW r1 oursmtp@BLOOMERY_RESIDENCY=mid-p33-s1 d=512 n=4 ctx=2048 \| tok/s\(mean\) 60\.61 @ .*\| residency mid-p33-s1 \(set\) passes 3 kept 30 ' &&
    want mtp-lever-arm "$L2" 1 '^mean oursmtp@BLOOMERY_RESIDENCY=mid-p33-s1 512 +60\.61 tok/s ' &&
    want mtp-lever-arm "$L2" 0 '^ratio ' &&
    want mtp-lever-arm "$L2" 0 '^(FAIL )?xcheck .*oursmtp' &&
    want mtp-lever-arm "$L2" 0 '^failed arms'; then
    pass mtp-lever-arm
  fi

  if [ "$RC3" != 1 ]; then
    fail lever-res-nolever "rc $RC3, want 1" "$L3"
  elif want lever-res-nolever "$L3" 1 "^FAIL r1 $LV d=512 rc=0 \\| the arm runs BLOOMERY_RESIDENCY=mid-p33-s1 and printed no residency lever record, which its schema declares \\| full output: " &&
    want lever-res-nolever "$L3" 1 '^ROW r1 ours d=512 ' &&
    want lever-res-nolever "$L3" 1 "^failed arms: r1:$LV@d=512$"; then
    pass lever-res-nolever
  fi

  L=$tmp/lever-res-none.log
  cp "$SCH" "$tmp/schema-glm5next.jsonl"
  python3 - "$tmp/schema-glm5next.jsonl" "$SCH" << 'PY'
import json, sys
src, out = sys.argv[1:]
lines = open(src).read().splitlines()
head = json.loads(lines[0])
kept = [l for l in lines[1:] if json.loads(l).get("kind") != "residency_lever"]
if len(kept) != len(lines) - 2:
    sys.exit(f"schema-glm5next: {len(lines) - 1 - len(kept)} residency_lever kinds, want 1")
head["kinds"] -= 1
open(out, "w").write("\n".join([json.dumps(head, separators=(",", ":"))] + kept) + "\n")
PY
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- 512 512@BLOOMERY_RESIDENCY=mid-p33-s1
  cp "$tmp/schema-glm5next.jsonl" "$SCH"
  if [ "$RC" != 1 ]; then
    fail lever-res-none "rc $RC, want 1" "$L"
  elif want lever-res-none "$L" 1 "^FAIL r1 $LV d=512 rc=0 \\| the arm runs BLOOMERY_RESIDENCY=mid-p33-s1, and generate_glm5next's checked-in schema \\(tools/bloomery/schema/generate_glm5next\\.jsonl\\) declares no residency record, so its row cannot carry them: " &&
    want lever-res-none "$L" 1 '^ROW r1 ours d=512 ' &&
    want lever-res-none "$L" 1 "^failed arms: r1:$LV@d=512$" &&
    want lever-res-none "$L" 1 "^\\[config\\] residency: generate_glm5next's checked-in schema declares no residency lever record: "; then
    pass lever-res-none
  fi

  L=$tmp/lever-levers.log
  : > "$tmp/tmp/stub-gen-env"
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" STUB_GEN_REFUSE=BLOOMERY_R8 -- 512 512@BLOOMERY_R8=off
  if [ "$RC" != 2 ]; then
    fail lever-levers "rc $RC, want 2" "$L"
  elif want lever-levers "$L" 1 '^depth-glm5next.sh: arm 512@BLOOMERY_R8=off: target/release/generate_glm5next refuses BLOOMERY_R8=off \(its --levers under them: Error: BLOOMERY_R8 is set, and generate_glm5next does not act on it \(stub\)\): the arm cannot run as written$' &&
    want lever-levers "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]|\[stub\] no lease)' &&
    want lever-levers "$tmp/tmp/stub-gen-env" 0 '.'; then
    pass lever-levers
  fi

  # The two-card mode (red on the runner before it, which has none: timing-card.sh refuses the mode and the
  # lever check refuses place=): a and bp arms in one run, each row the mode's card field and its place, the
  # a arm's load record the A6000 alone; the reference arm on the A6000 alone, its log one device.
  L=$tmp/twocard-place.log
  : > "$tmp/tmp/stub-bench-cvd"
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "${TC[@]}" BLOOMERY_GEN_PLACE=bp STUB_GEN_CARDS_ON=1 \
    STUB_BENCH_DEVLINES=1 -- 6 6@place=a lcpp27752:6
  if [ "$RC" != 0 ]; then
    fail twocard-place "rc $RC, want 0" "$L"
  elif [ "$(sort -u "$tmp/tmp/stub-bench-cvd")" != "$STUB_GPU_A6000" ]; then
    fail twocard-place "the reference bench saw CUDA_VISIBLE_DEVICES $(sort -u "$tmp/tmp/stub-bench-cvd" | paste -sd'|' -), want the A6000 alone ($STUB_GPU_A6000)" "$L"
  elif want twocard-place "$L" 1 '^ROW r1 ours d=6 n=4 ctx=2048 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place bp ' &&
    want twocard-place "$L" 1 '^ROW r1 ours@place=a d=6 n=4 ctx=2048 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place a ' &&
    want twocard-place "$L" 1 '^ROW r1 lcpp27752 d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 \| build stub \(0\) \| device NVIDIA RTX A6000 \(stub\) \| ' &&
    want twocard-place "$L" 1 '^ratio d=6 +ours/ours@place=a +mean ' &&
    want twocard-place "$L" 1 '^\[config\] placements: 6 bp, 6@place=a a \(an arm.s place= over BLOOMERY_GEN_PLACE=bp\)$' &&
    want twocard-place "$L" 1 '^\[config\] two cards: A6000\+3090, our arms at their placements; the reference arms on the A6000 alone \(CUDA_VISIBLE_DEVICES=GPU-0+-0000-0000-0000-0+\), no -ts arm for this model$' &&
    want twocard-place "$L" 8 '^    3090 cap: ok ' &&
    want twocard-place "$L" 8 '^    xid: 0 NVRM Xid line\(s\) since the lease was taken ' &&
    want twocard-place "$L" 1 '^ratio place d=6 +ours/ours@place=a +mean ' &&
    want twocard-place "$L" 2 '^ratio place ' &&
    want twocard-place "$L" 0 '^FAIL '; then
    pass twocard-place
  fi
  # One drafted arm at two placements: oursmtp and oursmtp@place=a paired by round in the placement table,
  # decode and prefill, bp over a; ours and ours@place=a too.
  L=$tmp/twocard-place-mtp.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 "${TC[@]}" BLOOMERY_GEN_PLACE=bp STUB_GEN_CARDS_ON=1 \
    STUB_GEN_MTP_REC=1 -- 6 6@place=a oursmtp:6 oursmtp:6@place=a
  if [ "$RC" != 0 ]; then
    fail twocard-place-mtp "rc $RC, want 0" "$L"
  elif want twocard-place-mtp "$L" 2 '^ROW r[12] oursmtp d=6 n=4 ctx=2048 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place bp ' &&
    want twocard-place-mtp "$L" 2 '^ROW r[12] oursmtp@place=a d=6 n=4 ctx=2048 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place a ' &&
    want twocard-place-mtp "$L" 1 '^ratio place d=6 +oursmtp/oursmtp@place=a +mean 1\.0000 ± 0\.0000 \(n=2\)  of means 1\.0000  per round: r1 1\.0000 r2 1\.0000$' &&
    want twocard-place-mtp "$L" 1 '^ratio place pp p=6 +oursmtp/oursmtp@place=a +mean ' &&
    want twocard-place-mtp "$L" 1 '^ratio place d=6 +ours/ours@place=a +mean ' &&
    want twocard-place-mtp "$L" 4 '^ratio place ' &&
    want twocard-place-mtp "$L" 1 '^ratio d=6 +ours/ours@place=a +mean ' &&
    want twocard-place-mtp "$L" 0 '^FAIL '; then
    pass twocard-place-mtp
  fi
  # A lever arm at two placements, both naming place= (BLOOMERY_GEN_PLACE unset), bp's steps faster (15 ms
  # against 16.5): bp / a = 1.1000 by round, in either order of the arms and of their items.
  LP=oursmtp@BLOOMERY_RESIDENCY=mid-p33-s1,place=bp LA=oursmtp@place=a,BLOOMERY_RESIDENCY=mid-p33-s1
  L=$tmp/twocard-place-order.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 "${TC[@]}" STUB_GEN_CARDS_ON=1 STUB_GEN_MTP_REC=1 \
    STUB_GEN_SMS_BP=15.0000 -- "oursmtp:512@${LA#oursmtp@}" "oursmtp:512@${LP#oursmtp@}"
  RC1=$RC
  L2=$tmp/twocard-place-order2.log
  stub_run "$L2" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 "${TC[@]}" STUB_GEN_CARDS_ON=1 STUB_GEN_MTP_REC=1 \
    STUB_GEN_SMS_BP=15.0000 -- "oursmtp:512@${LP#oursmtp@}" "oursmtp:512@${LA#oursmtp@}"
  if [ "$RC1" != 0 ] || [ "$RC" != 0 ]; then
    fail twocard-place-order "rc $RC1 (a first) and $RC (bp first), want 0 and 0" "$L"
  elif want twocard-place-order "$L" 2 "^ROW r[12] $LP d=512 .*\| place bp " &&
    want twocard-place-order "$L" 2 "^ROW r[12] $LA d=512 .*\| place a " &&
    want twocard-place-order "$L" 1 "^ratio place d=512 +$LP/$LA +mean 1\.1000 ± 0\.0000 \(n=2\)  of means 1\.1000  per round: r1 1\.1000 r2 1\.1000$" &&
    want twocard-place-order "$L2" 1 "^ratio place d=512 +$LP/$LA +mean 1\.1000 ± 0\.0000 \(n=2\)  of means 1\.1000  per round: r1 1\.1000 r2 1\.1000$" &&
    want twocard-place-order "$L" 1 "^ratio place pp p=512 +$LP/$LA +mean " &&
    want twocard-place-order "$L" 2 '^ratio place ' &&
    want twocard-place-order "$L2" 2 '^ratio place ' &&
    want twocard-place-order "$L" 0 '^FAIL ' &&
    want twocard-place-order "$L2" 0 '^FAIL '; then
    pass twocard-place-order
  fi
  # An a arm whose load record names the 3090: a FAIL row naming the cards.
  L=$tmp/twocard-place-3090.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "${TC[@]}" STUB_GEN_CARDS_ON=1 \
    'STUB_GEN_CARDS_A=[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090]' -- 6@place=a
  if [ "$RC" != 1 ]; then
    fail twocard-place-3090 "rc $RC, want 1" "$L"
  elif want twocard-place-3090 "$L" 1 "^FAIL r1 ours@place=a d=6 rc=0 \| two cards: the engine.s load record names cards \[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090\], and --place a loads the A6000 alone: the 3090 must stay idle \| full output: " &&
    want twocard-place-3090 "$L" 0 '^ROW '; then
    pass twocard-place-3090
  fi
  # generate_glm5next's load record as it is today, with no cards field: every ours row a FAIL row naming it.
  L=$tmp/twocard-nocards.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "${TC[@]}" -- 6
  if [ "$RC" != 1 ]; then
    fail twocard-nocards "rc $RC, want 1" "$L"
  elif want twocard-nocards "$L" 1 "^FAIL r1 ours d=6 rc=0 \| two cards: the engine.s load record names no cards: which cards it loaded cannot be told \| full output: " &&
    want twocard-nocards "$L" 0 '^ROW '; then
    pass twocard-nocards
  fi
  # The dry run: each ours arm's --place, the reference's CUDA_VISIBLE_DEVICES, the precheck, the placements.
  L=$tmp/twocard-place-dry.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 "${TC[@]}" BLOOMERY_GEN_PLACE=bp -- 6 6@place=a lcpp27752:6
  if [ "$RC" != 0 ]; then
    fail twocard-place-dry "rc $RC, want 0" "$L"
  elif want twocard-place-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50005> -n 4 --ctx 2048 --place bp --time $' &&
    want twocard-place-dry "$L" 1 '^\[dry\]     timeout --kill-after=10 60 target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\.\.50005> -n 4 --ctx 2048 --place a --time $' &&
    want twocard-place-dry "$L" 1 "^\[dry\]     timeout --kill-after=10 60 env CUDA_VISIBLE_DEVICES=$STUB_GPU_A6000 [^ ]*/pr27752/llama-bench -m " &&
    want twocard-place-dry "$L" 1 '^\[dry\] two-card precheck: ok$' &&
    want twocard-place-dry "$L" 1 '^\[dry\] placements: 6 bp, 6@place=a a \(an arm.s place= over BLOOMERY_GEN_PLACE=bp\)$'; then
    pass twocard-place-dry
  fi
  # The aggregate arm (the runner's <D>@BLOOMERY_GEN_SLOTS=N) beside its plain twin, both under one residency
  # word: the plain arm at 33 ms a step (30.30 tok/s), the 2-slot arm at 44 ms a round of 2 positions over its 3
  # rounds (2 · 3 · 1000 / 132 = 45.45 tok/s): 45.45 / 30.30 = 1.5000 in each round.
  RW=BLOOMERY_RESIDENCY=mid-p0-s1
  SL=ours@BLOOMERY_GEN_SLOTS=2,$RW TW=ours@$RW
  L=$tmp/slots.log
  : > "$tmp/tmp/stub-gen-slots"
  stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- "6@$RW" "6@BLOOMERY_GEN_SLOTS=2,$RW"
  if [ "$RC" != 0 ]; then
    fail slots "rc $RC, want 0" "$L"
  elif [ "$(paste -sd' ' "$tmp/tmp/stub-gen-slots")" != "slots=none depth=6 slots=2 depth=12 slots=2 depth=12 slots=none depth=6" ]; then
    fail slots "the lever and the feed per process, want the twin's none of 6 ids and the aggregate arm's 2 of 12, in the rotation's order" "$tmp/tmp/stub-gen-slots"
  elif want slots "$L" 2 "^ROW r[12] $SL d=6 n=4 ctx=2048 \\| tok/s\\(aggregate\\) 45\\.45 @ n=2·3, depth 6, A6000 \\(stub\\) \\| place a card_experts 2627 host_experts 9469 \\| prompt ids 50000\\.\\.50011 \\| slots 2 \\| tok/s\\(per stream, mean\\) 22\\.73 \\| p50 44\\.0000 ms/pass \\| mean 44\\.0000 ms/pass \\| warm 0 \\| first10_p50 44\\.0000 \\| last10_p50 44\\.0000 \\| distinct_tokens 6 \\| pp_tok/s 60\\.00 \\(n=6, passes=1, kind=batch\\) \\| residency mid-p0-s1 \\(set\\) passes 0 " &&
    want slots "$L" 2 "^ROW r[12] $TW d=6 n=4 ctx=2048 \\| tok/s\\(mean\\) 30\\.30 @ n=4, depth 6, .*\\| prompt ids 50000\\.\\.50005 \\| " &&
    want slots "$L" 1 "^mean $SL 6 +45\\.45 tok/s  \\[45\\.45\\.\\.45\\.45, spread 0\\.00%\\]  \\(aggregate of 2 slots\\) \\(n=2\\)$" &&
    want slots "$L" 1 "^mean $TW 6 +30\\.30 tok/s  \\[30\\.30\\.\\.30\\.30, spread 0\\.00%\\]  \\(n=2\\)$" &&
    want slots "$L" 1 '^ratio slots ' &&
    want slots "$L" 1 "^ratio slots d=6 +$SL/$TW +mean 1\\.5000 ± 0\\.0000 \\(n=2\\)  of means 1\\.5000  per round: r1 1\\.5000 r2 1\\.5000$" &&
    want slots "$L" 0 '^ratio d=6 .*GEN_SLOTS' &&
    want slots "$L" 0 '^failed arms'; then
    pass slots
  fi
  # The residency off (generate_glm5next's unset): no residency record, no pass for the residency clause to read,
  # so the row stands on its time pass records alone.
  L=$tmp/slots-res-off.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" -- 6@BLOOMERY_GEN_SLOTS=2
  if [ "$RC" != 0 ]; then
    fail slots-res-off "rc $RC, want 0" "$L"
  elif want slots-res-off "$L" 1 "^ROW r1 ours@BLOOMERY_GEN_SLOTS=2 d=6 n=4 ctx=2048 \\| tok/s\\(aggregate\\) 45\\.45 @ n=2·3, depth 6, A6000 \\(stub\\) \\| place a card_experts 2627 host_experts 9469 \\| prompt ids 50000\\.\\.50011 \\| slots 2 \\| tok/s\\(per stream, mean\\) 22\\.73 \\| p50 44\\.0000 ms/pass \\| mean 44\\.0000 ms/pass \\| .*\\| pp_tok/s 60\\.00 \\(n=6, passes=1, kind=batch\\) \\| " &&
    want slots-res-off "$L" 0 '^ROW .*\| residency ' &&
    want slots-res-off "$L" 0 '^FAIL '; then
    pass slots-res-off
  fi
  # A FAIL row for each clause: a round of 1 position, kind=slots records from a plain arm, an aggregate arm that
  # printed the plain lines, a step pass in place of the pass.
  slots_fail() { # slots_fail <name> <pattern> <env…> -- <arm>
    local name=$1 pat=$2 e=()
    shift 2
    while [ "$1" != -- ]; do e+=("$1"); shift; done
    shift
    L=$tmp/$name.log
    stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 BLOOMERY_TIMING_GPU="$GPU_A" ${e[@]+"${e[@]}"} -- "$@"
    if [ "$RC" != 1 ]; then
      fail "$name" "rc $RC, want 1" "$L"
    elif want "$name" "$L" 1 "$pat" && want "$name" "$L" 0 '^ROW '; then
      pass "$name"
    fi
  }
  slots_fail slots-pos "^FAIL r1 $SL d=6 rc=0 \\| 3 of its 3 counted kind=slots records hold positions other than its 2 slots \\| " STUB_GEN_SLOTS_POS=1 -- "6@BLOOMERY_GEN_SLOTS=2,$RW"
  slots_fail slots-leak '^FAIL r1 ours d=6 rc=0 \| its output holds 3 time pass record\(s\) of kind=slots and the arm names no BLOOMERY_GEN_SLOTS: ' STUB_GEN_SLOTS_LEAK=2 -- 6
  slots_fail slots-none "^FAIL r1 $SL d=6 rc=0 \\| the arm runs BLOOMERY_GEN_SLOTS=2 and printed no counted time pass record of kind=slots: " STUB_GEN_SLOTS_NOPASS=1 -- "6@BLOOMERY_GEN_SLOTS=2,$RW"
  slots_fail slots-res-step "^FAIL r1 $SL d=6 rc=residency \\| the slots residency clause: the residency pass at boundary=3 reads pass=step kept=2 before its first pass=slots, where only its 2 slots. prompt calls \\(pass=prompt kept=0\\) stand \\| " STUB_GEN_SLOTS_RESPASS=step -- "6@BLOOMERY_GEN_SLOTS=2,$RW"
  # N·D ids past the prose file: refused before the lease, where the plain arm of that D fits.
  PD=$(($(wc -l < "$PROSE_SRC") - 50000))
  L=$tmp/slots-past.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_GPU="$GPU_A" BLOOMERY_GEN_CTX=$((PD + 8)) -- "$PD" "$PD@BLOOMERY_GEN_SLOTS=2,$RW"
  if [ "$RC" != 64 ]; then
    fail slots-past "rc $RC, want 64" "$L"
  elif want slots-past "$L" 1 "^depth-glm5next.sh: arm $PD@BLOOMERY_GEN_SLOTS=2,$RW: .* holds fewer than GLM_PROSE_FROM \\+ $((2 * PD)) = $((50000 + 2 * PD)) ids \\(2 slots × $PD\\)$" &&
    want slots-past "$L" 0 'holds fewer than GLM_PROSE_FROM \+ '"$PD"' ' &&
    want slots-past "$L" 0 '^(ROW|WARMUP|FAIL|\[config\]) '; then
    pass slots-past
  fi
  # The dry run: the aggregate arm's facts and its command line, its N·D ids under its variables.
  L=$tmp/slots-dry.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 BLOOMERY_TIMING_GPU="$GPU_A" -- "6@$RW" "6@BLOOMERY_GEN_SLOTS=2,$RW"
  if [ "$RC" != 0 ]; then
    fail slots-dry "rc $RC, want 0" "$L"
  elif want slots-dry "$L" 1 "^\\[dry\\] 6@BLOOMERY_GEN_SLOTS=2,$RW: slots=2 feed=12 twin=$TW$" &&
    want slots-dry "$L" 1 "^\\[dry\\]     timeout --kill-after=10 60 env BLOOMERY_GEN_SLOTS=2 $RW target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\\.\\.50011> -n 4 --ctx 2048 --place a --time $" &&
    want slots-dry "$L" 1 "^\\[dry\\] 6@$RW:$" &&
    want slots-dry "$L" 1 "^\\[dry\\]     timeout --kill-after=10 60 env $RW target/release/generate_glm5next --tokens <GLM_PROSE ids 50000\\.\\.50005> -n 4 --ctx 2048 --place a --time $"; then
    pass slots-dry
  fi
else
  echo "skip ours-timed, cold-retry, srv, srv-xcheck, srv-ctx, srv-flags, srv-dry, srv-probe, srv-fail, mtp, mtp-rec, mtp-lever, lever-pair, mtp-lever-arm, lever-res-nolever, lever-res-none, lever-levers, twocard-place, twocard-place-mtp, twocard-place-order, twocard-place-3090, twocard-nocards, twocard-place-dry, slots, slots-res-off, slots-pos, slots-leak, slots-none, slots-res-step, slots-past and slots-dry: no prose ids at $PROSE_SRC (the profile pins their sha256)"
fi

# same_view: a dry run less the lines this runner changes on purpose (dry-same).
same_view() {
  grep -vE '^\[dry\] (preheat|prompt|WARMUP r0|server arms):|^\[dry\] lcpp2775[24](srv|mtp)|llama-server' |
    sed -E 's/--tokens <[^>]*>/--tokens <prompt>/; s/row "POST \/completion: [^,]*,/row "POST \/completion: <prompt>,/'
}
if [ -n "$BASE" ]; then
  SAME_ARMS=(6 lcpp27752:6 lcpp27754:6 lcpp27752pp:4 lcpp27754pp:4 lcpp27752pp8:4 lcpp27754pp4096:4 lcpp27752srv:6 lcpp27754srv:6 lcpp27752mtp:6 lcpp27754mtp:6 exl3:256 exl3pp:256)
  cp "$BASE" "$T/tools/ref/depth-glm5next-base.sh"
  L=$tmp/dry-same-base.log L2=$tmp/dry-same.log
  # The base runner's server arms read four profile strings this profile no longer holds (their lines are
  # left out of the comparison): given here, so the base runs.
  RUNNER_FILE=tools/ref/depth-glm5next-base.sh stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 \
    LCPP27752_SRV_FLAGS=base LCPP27754_SRV_FLAGS=base LCPP27752_MTP_FLAGS=base LCPP27754_MTP_FLAGS=base -- "${SAME_ARMS[@]}"
  RC1=$RC
  stub_run "$L2" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DRY=1 -- "${SAME_ARMS[@]}"
  if [ "$RC1" != 0 ] || [ "$RC" != 0 ]; then
    fail dry-same "rc $RC1 (base) and $RC (tested), want 0 and 0" "$L2"
  elif ! diff <(same_view < "$L") <(same_view < "$L2") > "$tmp/dry-same.diff"; then
    fail dry-same "the dry runs differ, less the lines changed on purpose (base <, tested >)" "$tmp/dry-same.diff"
  elif want dry-same "$L2" 13 '^\[dry\]     '; then
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
