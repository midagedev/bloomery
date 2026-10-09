#!/usr/bin/env bash
# The depth-qwen3moe.sh stub test: the runner's arm loop with no lease, no card and no model. It copies the
# runner (DEPTH_QWEN3MOE_RUNNER, default this tree's) into a fresh temporary tree beside this tree's
# timing-card.sh, lease-probe.sh, tdist.py, load-groups.sh, lcpp-fit.sh, cold-blocks.sh, lever-arms.sh,
# arm-place.sh, slots-arm.sh, drive-read.sh and lcpp-warm.sh, and a copy of lease.sh whose lease_take is replaced by a line that takes nothing; cards.sh
# there is depth-stub-cards.sh's, two made-up UUIDs;
# ref-paths.sh there is a stub qwen4exp profile whose engines are stub scripts (llama-bench as lcpp and ik,
# llama-server, generate_qwen3moe, mistralrs, nvidia-smi). Nothing it starts loads a model or touches a
# card. The fault counter every copy reads is a file the stub engines add to when a case makes them fault
# (depth-ds41-stub.sh's). depth-ds41-stub.sh is the
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
#                the whole process — and the ours and bin: rows then carry ` | drive_read_bytes <n> (<dev>)`
#                before the cold tag, the others no such column; the closing summary counts the cold rows
#                and the means line carries the cold and probe-off counts.
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
# The server arms on the prose ids, the placement, MTP's acceptance and the cross-check (red on the runner
# before them: arm usage or refusal, rc 64; no place, E(4) or xcheck line):
#   srv-prose    prose:512 lcppsrv:prose:512, one round: the server row `lcppsrv@prose` fed the corpus's
#                first 512 ids (twice: the warm-up and the timed request), its -c ours' 768, its ratio in the
#                prose table and not in the lcg one, its decode row's prompt rate in the prose prefill table,
#                `xcheck prose p=512 r1 ours@prose(r1)/lcppsrv@prose: same 4`, and `cpu-busy rows: 0 of 2`.
#   srv-xcheck   the same with ours' token 0 another id (STUB_GEN_TOKEN0=7): the FAIL xcheck line, in the
#                failed arms, the server label dropped from the ratios, rc 1; the server's second id another
#                (STUB_SRV_TOKEN1=7): a [xcheck-tail] line, the ratio kept, rc 0.
#   place-gate   BLOOMERY_GEN_PLACE=gate with the 3090 as the timing card, 6: `--place gate` on the stub's
#                command line, the row's `place gate`; the load line naming place=a (STUB_GEN_PLACE_RAN): a
#                FAIL row naming both, rc 1; gate with the A6000 as the timing card: refused by name, rc 64.
#   mtp          STUB_GEN_MTP=1, 6: the row's `mtp E(4) 2.500 = positions 10 / passes 4, kept [1, 1, 1, 1]` and
#                its pass time from the SMOKE line, `ms/pass 12.5000 (mean_ms × steps 10 / passes 4)`.
#   mtp-pass-ratio  STUB_GEN_MTP=1, 6 6@STUB_GEN_PASSES=5, two rounds: one `mean pass` line a label (12.5 and
#                10.0 ms a pass) and `ratio pass d=6 ours/ours@STUB_GEN_PASSES=5 mean 0.8000`, the passes a second
#                80 / 100, beside the decode ratio 1.0000 the two arms' equal tok/s give.
#   mtp-pass-bad STUB_GEN_MTP=1 STUB_GEN_PASSES=0, 6: a SMOKE line with passes=0 is a FAIL row naming it, rc 1.
# The aggregate arm (the runner's <D>@BLOOMERY_GEN_SLOTS=N; red on the runner before it, which refuses the
# name as no registry row or reads the arm as a plain lever arm):
#   slots        6 6@BLOOMERY_GEN_SLOTS=2, two rounds: the aggregate arm fed 12 ids in a load of its own, its
#                row `tok/s(aggregate) 250.00 @ n=2·3` (2 positions a round, 3 rounds at 8 ms), its means line
#                beside the plain one's `(aggregate of 2 slots)`, one `ratio slots d=6
#                ours@BLOOMERY_GEN_SLOTS=2/ours mean 1.2500` and no `ratio d=` line for the pair; rc 0.
#   slots-pos    a round of 1 position in the 2-slot arm (STUB_GEN_SLOTS_POS=1): a FAIL row naming it, rc 1.
#   slots-leak   kind=slots records from the plain arm 6 (STUB_GEN_SLOTS_LEAK=2): a FAIL row naming it, rc 1.
#   slots-none   the 2-slot arm printing the plain lines (STUB_GEN_SLOTS_NOPASS=1, a binary that ignored the
#                lever): a FAIL row naming the missing records, rc 1.
#   bin-slots    bin:<stub base>:prose:3@BLOOMERY_GEN_SLOTS=2 prose:3@BLOOMERY_GEN_SLOTS=2, two rounds: the bin:
#                arm an aggregate arm as ours' is, fed the corpus's first 6 ids (every process 6 ids), no FAIL row,
#                both rows `tok/s(aggregate) 250.00` with `slots 2`, the bin: label in the slots table (no plain
#                twin) and one `ratio slots bin d=3 ours@prose@BLOOMERY_GEN_SLOTS=2/bin:base@prose@…` line, mean
#                1.0000; rc 0. Red on the runner before it, which fed the bin: arm 3 ids and read its kind=slots
#                records as a FAIL row (an arm that names no N), rc 1.
# The capped arms (the runner's Mem; red on the runner before them, which refused a bin: arm's mem= by name and
# evicted nothing). A stub systemd-run in the tree's PATH starts no scope: it records its -p items and runs the
# command. The two-shard split set evict and evict-held read sits under this tree's target/ (disk-backed on the
# box; a tmpfs one fails both cases by name, its pages being ones posix_fadvise does not evict):
#   bin-mem      bin:<stub base>:6@mem=29G 6, two rounds: two systemd-run calls `MemoryMax=29G MemorySwapMax=0` (the
#                bin: arm's processes), its rows `capped mem=29G` after the card field, the scope's witness line, the
#                evict line over the profile's file (empty: pages=0 resident before=0 after=0), the plain arm's rows
#                uncapped; rc 0.
#   evict        bin:<stub base>:6@mem=29G,model=<set> 6@mem=29G,model=<set>, two rounds, each process reading the
#                set back in (STUB_GEN_READ_MODEL=1): four evict lines `shards=2 pages=2·SP resident before=2·SP
#                after=0`, every row capped; rc 0.
#   evict-held   a holder process with the first shard mapped and every page touched, 6@mem=29G,model=<set>, one
#                round: the evict line's `after=SP`, `FAIL r1 … rc=evict | evict: SP of the 2·SP pages …`, no process
#                started, no row; rc 1.
#   res-sums     generate_qwen3moe's schema with the residency kinds (the stub tree's copy, its own residency
#                rows replaced by generate_ds41's), STUB_GEN_RES=mid-p148-s1, 6@BLOOMERY_RESIDENCY=mid-p148-s1: the row's
#                `residency mid-p148-s1 (set) passes 3 kept 30 landed 3 late 1 made 3 bytes 12288` (the none and
#                prompt boundaries not counted) and the per-arm mean line; STUB_GEN_STATS=1 adds `host slots/token
#                3.4`. Under a copy with no residency kind the same arm is a FAIL row naming the schema, rc 1.
#   res-nolever  the copy with the residency kinds, 6@BLOOMERY_RESIDENCY=mid-p148-s1 with no STUB_GEN_RES (the stub
#                prints no lever record): a FAIL row naming the missing record, rc 1, never a row with no column.
#   cpu-guard    STUB_CPU_BUSY=1 (every CPU sample 99 %), 6 lcpp:6: both rows [cpu-busy], `cpu-busy rows: 2 of 2`.
#   profile-3090 the real models/qwen4exp.sh: -ncmoe 43 with the 3090 as the timing card, 26 with the A6000 or
#                none; another card with no 3090 UUID resolved: refused by name, rc 64.
#                DEPTH_QWEN3MOE_PROFILE names another copy of the profile (FAIL-first).
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
# The two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090, timing-card.sh), red on the runner before it (it
# has no mode: timing-card.sh leaves it no card, and its rows would name one):
#   twocard      lcpp:6 lcpppp:4 lcppfit:6, one round: every row's card field `A6000+3090` and device
#                field both cards, the stub llama-bench given -ts 1.5/1.5 (the fit arm dropping it), the
#                witness's two-card lines (each card, the 3090's cap, the Xid count), the [config] line; rc 0.
#   twocard-dry  the same under BLOOMERY_DRY=1: both cards visible, the precheck's lines and `ok`, the lcpp
#                command line with -ts.
#   twocard-srv  lcppsrv:6 lcppsrvpp:4, one round: the server arms in the mode, the profile's -ts in each
#                server's command line, each row `A6000+3090` with both cards as its device, no server left up;
#                rc 0 (depth-ds41-stub.sh has the one-card and Xid failures of a server arm).
#   twocard-arms[-bin|-ik|-mrs]  6@place=gate, bin:<base>:6 under BLOOMERY_GEN_PLACE=gate, ik:6 and mrs:6
#                each refused by name before anything runs (rc 64): the gate plan loads the 3090 alone, ik
#                and mistral.rs have no two-card line.
#   twocard-place  BLOOMERY_GEN_PLACE=bp, 6 6@place=a lcpp:6, one round, the stub naming its load's cards
#                (STUB_GEN_CARDS_ON): each placement its own load (--place bp, --place a), the ours rows
#                `place bp` and `place a` with the mode's card field, the [config] placements line; rc 0.
#   twocard-place-3090  an a arm whose load line names the 3090 (STUB_GEN_CARDS_A): its FAIL row; rc 1.
#   twocard-nocards  a load line with no cards= field (a binary from before generate_qwen3moe named its
#                cards): the FAIL row; rc 1.
#   twocard-place-dry  the dry run: each arm's --place and load key, the placements line.
# An arm's place= under one card (red on the runner before it, which reads place= as a lever and refuses
# it as no registry row, rc 64 with another text):
#   bp-onecard-arm, place-ref, place-word, place-twice, place-card-arm  6@place=bp, lcpp:6@place=a, 6@place=b2,
#                6@place=a,place=a and 6@place=gate with the A6000 the timing card: each refused by name, rc 64.
#   place-one    6 6@place=a: two rows at `place a` (one load: the same key), no variable named place in
#                the binary's environment.
#   twocard-profile  a profile with no two-card line (TWO_CARD_PLACEMENT empty): refused by name, rc 64.
#   twocard-cap  the 3090 at 300 W: refused before the lease, rc 78, no row.
#   twocard-gone the 3090 not answering nvidia-smi: refused before the lease, rc 69.
#   twocard-lease a lease.sh with no two-card record: refused before the lease, rc 64.
#   twocard-value BLOOMERY_TIMING_CARDS=both: refused by name, rc 64.
#   twocard-xid  lcpp:6 lcpppp:4 lcpp:5, an Xid 79 line in the kernel journal during lcpppp:4: its FAIL row
#                naming the Xid, the arms before and after it rows (the count moves on), rc 1.
#   twocard-one  a llama-bench that sees one CUDA device (lcpp:6): its FAIL row naming it, rc 1.
#   twocard-busy a compute process on the 3090 as the first arm starts, gone at the next poll: the
#                [cards-busy] wait and then the row; rc 0.
#   twocard-optin  timing-card.sh sourced by a runner that does not opt in: TIMING_GPU and
#                CUDA_VISIBLE_DEVICES empty and the refusal named (lease_take refuses an empty TIMING_GPU).
# The server arms and the warm rows (lcpp-warm.sh; red on the runner before them):
#   srv          6 lcppsrv:6 lcppsrvpp:4 lcppsrvpp8:4 lcppsrvfit:6, one round: the server rows in this runner's
#                field order (the tags and the majflt column after the wall), the flags of the qwen4exp
#                profile in the server's spellings (-lzm off kept), the fit twin's column, ratio rows.
#   srv-exit     a server that exits before it listens: a FAIL row, rc 1.
#   srv-cold     BLOOMERY_WARM_ROWS=1, the server's timed request faulting once: COLD, then the retry's row.
#   warm         BLOOMERY_WARM_ROWS=1, 6 4 lcpp:6: the two ours arms in one load (one C) as 6 6 4 4, each
#                after its PRIME; 4's row cold once (a fresh load of 4 4, then clean), lcpp:6 cold twice (COLD,
#                then FAIL rc=cold); rc 1.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNNER=${DEPTH_QWEN3MOE_RUNNER:-$ROOT/tools/ref/depth-qwen3moe.sh}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/depth-qwen3moe-stub.XXXXXX")
# On any exit: evict-held's holder stopped (its pid from its own file) and the capped arms' shard directory
# (SH, below) removed with the temporary tree.
trap '[ ! -s "$tmp/holder.pid" ] || kill "$(cat "$tmp/holder.pid")" 2> /dev/null; rm -rf "$tmp" ${SH:+"$SH"}' EXIT
T=$tmp/tree
mkdir -p "$T/tools/ref" "$T/tools/bloomery" "$T/target/release" "$T/base/target/release" "$T/bin" "$tmp/tmp"
cp "$RUNNER" "$T/tools/ref/depth-qwen3moe.sh"
cp "$ROOT/tools/ref/timing-card.sh" "$ROOT/tools/ref/lease-probe.sh" \
  "$ROOT/tools/ref/lease.sh" "$ROOT/tools/ref/tdist.py" "$ROOT/tools/ref/load-groups.sh" \
  "$ROOT/tools/ref/lcpp-fit.sh" "$ROOT/tools/ref/cold-blocks.sh" "$ROOT/tools/ref/lever-arms.sh" \
  "$ROOT/tools/ref/arm-place.sh" "$ROOT/tools/ref/slots-arm.sh" "$ROOT/tools/ref/drive-read.sh" \
  "$T/tools/ref/"
[ ! -f "$ROOT/tools/ref/lcpp-warm.sh" ] || cp "$ROOT/tools/ref/lcpp-warm.sh" "$T/tools/ref/"
# records.py and the checked-in schemas: the runner reads the arm's residency, mtp and stat records by kind.
cp -R "$ROOT/tools/bloomery/records.py" "$ROOT/tools/bloomery/schema" "$T/tools/bloomery/"
# The lever registry, which an arm's @NAME=VALUE list is checked against (before the stub binaries are
# written, so they are newer than every crates/ source).
mkdir -p "$T/crates/levers/src"
cp "$ROOT/crates/levers/src/registry.rs" "$T/crates/levers/src/"
# The copy's one row of its own, STUB_GEN_PASSES (the stub engine's pass count): a lever arm sets a registry
# row, and the mtp-pass-ratio case's lever arm sets this one.
echo '// LeverSpec { name: "STUB_GEN_PASSES", site: Site::Parsed } (the stub test'"'"'s own row)' >> "$T/crates/levers/src/registry.rs"
echo 'lease_take() { echo "[stub] no lease: the stub test'"'"'s copy of lease.sh takes nothing"; }' >> "$T/tools/ref/lease.sh"
# STUB_CPU_BUSY=1: every CPU sample reads 99 % of one cpu from a stub process (the cpu-guard case).
# shellcheck disable=SC2016 # written into the stub tree's lease.sh, expanded there
echo '[ -z "${STUB_CPU_BUSY:-}" ] || cpu_busy_sample() { CPU_BUSY_READING="99.0 stubproc 99.0;" CPU_BUSY_OWN= CPU_BUSY_ERR= CPU_BUSY_SPAN=1; }' >> "$T/tools/ref/lease.sh"
# The stub's lease writes no record; it says it would write a two-card one (the precheck asks), and
# STUB_ONE_CARD_LEASE=1 takes that back: a lease.sh without the two-card record.
# shellcheck disable=SC2016 # the line is written for the copy to expand
echo 'if [ -n "${STUB_ONE_CARD_LEASE:-}" ]; then unset LEASE_CARDS_RECORD; else LEASE_CARDS_RECORD=1; fi' >> "$T/tools/ref/lease.sh"
# The fault counter: $STUB_MAJFLT, a number the stub engines add to.
# shellcheck disable=SC2016 # the copies expand them when they run
{
  echo 'majflt_now() { cat "$STUB_MAJFLT"; }'
  echo 'majflt_mark() { awk -v f="$1" -v re="$2" -v src="$STUB_MAJFLT" '"'"'!s && re != "" && $0 ~ re { getline l < src; close(src); print l > f; close(f); s = 1 } { print; fflush() }'"'"'; }'
} >> "$T/tools/ref/cold-blocks.sh"
echo 'lg_majflt() { cat "$STUB_MAJFLT"; }' >> "$T/tools/ref/load-groups.sh"
[ ! -f "$T/tools/ref/lcpp-warm.sh" ] || echo 'lcpp_srv_majflt() { cat "$STUB_MAJFLT"; }' >> "$T/tools/ref/lcpp-warm.sh"
touch "$T/model-00001-of-00001.gguf"
cat > "$T/tools/ref/ref-paths.sh" << EOF
# shellcheck shell=bash
MODEL_NAME=qwen4exp
MODEL=$T/model-00001-of-00001.gguf
IK=$T IKBIN=$T/bin/ik-bench LCPP=$T LCPPBIN=$T/bin/lcpp-bench MRS=$T MRSBIN=$T/bin/mistralrs
IK_GPU_FLAGS='-ngl 99 --n-cpu-moe 1' IK_GPU_DEFAULT_FLAGS='-ngl 99 --n-cpu-moe 2'
LCPP_GPU_FLAGS='-ngl 99 -fa on -ncmoe 1'
MRS_FLAGS='--format gguf'
TWO_CARD_PLACEMENT=
if [ "\${BLOOMERY_TIMING_CARDS:-}" = a6000+3090 ] && [ -z "\${STUB_NO_TWO_CARD:-}" ]; then
  LCPP_GPU_FLAGS='-ngl 99 -fa on -ncmoe 1 -ts 1.5/1.5'
  TWO_CARD_PLACEMENT='the stub two-card line: -ncmoe 1 -ts 1.5/1.5'
fi
EOF
# shellcheck source=tools/ref/depth-stub-cards.sh
. "$HERE/depth-stub-cards.sh"
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
eng=${0##*/} k='' p=0 n=0 d='' gp='' prog='' fitt='' verb='' ts=''
while [ $# -gt 0 ]; do
  case $1 in
    --n-cpu-moe | -ncmoe) k=$2; shift ;; -p) p=$2; shift ;; -n) n=$2; shift ;; -d) d=$2; shift ;; -gp) gp=$2; shift ;;
    --progress) prog=1 ;; -fitt) fitt=$2; shift ;; -v) verb=1 ;; -ts) ts=$2; shift ;;
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
. "${STUB_BENCH_CARDS:?}"
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
# STUB_COLD_BENCH=<label>:<K>: the first K tests of that label add 100000 faults to the counter, 0.3 s
# after their timed window's progress line.
if [ "${STUB_COLD_BENCH%%:*}" = "$label" ]; then
  cm=${TMPDIR:-/tmp}/stub-cold-bench-${label//[^A-Za-z0-9]/_}
  c=$(cat "$cm" 2> /dev/null || echo 0)
  if [ "$c" -lt "${STUB_COLD_BENCH#*:}" ]; then
    echo $((c + 1)) > "$cm"
    sleep 0.3
    echo $(($(cat "$STUB_MAJFLT") + 100000)) > "$STUB_MAJFLT"
  fi
fi
if once ABORT; then
  echo "CUDA error: an illegal memory access was encountered (stub)" >&2
  echo "ggml_cuda_error: in function ggml_backend_cuda_graph_compute (stub)" >&2
  exit 134
fi
echo "| model | size | test | t/s |"
echo "| --- | ---: | ---: | ---: |"
once NOVAL || echo "| stub k=$k${ts:+ ts=$ts} | 1 | $label | $val ± 0.01 |"
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
# STUB_GEN_FAIL_DEPTH ends the process at rc 3 the first time (a marker file), after its prompt ids. Its
# load line names --place's placement (a when none; STUB_GEN_PLACE_RAN in its stead), each arm ends with
# its `tokens` line (token 0 STUB_GEN_TOKEN0, 1000 by default, then the step lines' ids), and under
# STUB_GEN_MTP an `mtp summary` of 10 positions in 4 passes and the MTP form of the SMOKE line (steps= the
# summary's positions, passes= its passes, no seeded=): STUB_GEN_PASSES=5 is 10 positions in 5 passes, 0 none
# in none, any other value an error. Every process appends its --place to $TMPDIR/stub-gen-place (`-` when
# none). A variable named `place` or `mem` in its environment ends it at rc 9; under STUB_GEN_READ_MODEL=1 it
# first reads every shard of BLOOMERY_REF_MODEL's split set. Its load line names no cards unless
# STUB_GEN_CARDS_ON=1 (twocard-nocards keeps a load line without them); under it the line names the cards
# its --place loads, as generate_qwen3moe's does (`cards=`: the A6000 under a or none, STUB_GEN_CARDS_A in
# their stead; the 3090 under gate; both under bp).
# Under BLOOMERY_GEN_SLOTS=N (N >= 2, or STUB_GEN_SLOTS_LEAK=N on any arm) each arm is N slots of depth / N
# ids: their step, time prompt and tokens lines tagged `slot=<j>`, a `capture slots=` line, each round a
# `time pass … kind=slots` at 8 ms of N positions (STUB_GEN_SLOTS_POS in its stead), and the slots' SMOKE
# footer; STUB_GEN_SLOTS_NOPASS=1 prints the plain lines instead.
cat > "$T/target/release/generate_qwen3moe" << 'EOF'
#!/usr/bin/env bash
for v in place mem; do
  if printenv "$v" > /dev/null; then
    echo "error: a variable named $v reached the binary ($v=$(printenv "$v")): the runner passed an arm's $v= item as a variable" >&2
    exit 9
  fi
done
# STUB_GEN_READ_MODEL=1: the process reads every shard of BLOOMERY_REF_MODEL's split set, as a load faults them in.
if [ -n "${STUB_GEN_READ_MODEL:-}" ]; then
  m=${BLOOMERY_REF_MODEL:?STUB_GEN_READ_MODEL reads BLOOMERY_REF_MODEL}
  cat "${m%-0*-of-*}"-*-of-"${m##*-of-}" > /dev/null || exit 8
fi
tokens='' n=32 ctx=0 sync='' arms=() place=''
while [ $# -gt 0 ]; do
  case $1 in
    --tokens) tokens=$2; shift ;; -n) n=$2; shift ;; --ctx) ctx=$2; shift ;; --warm) shift ;;
    --arm) arms+=("$2"); shift ;; --arm-sync) sync=1 ;; --place) place=$2; shift ;;
  esac
  shift
done
echo "${place:--}" >> "${TMPDIR:-/tmp}/stub-gen-place"
count() { echo "$1" | tr ',' '\n' | grep -c .; }
listed=1
if [ ${#arms[@]} -eq 0 ]; then listed='' arms=("$tokens"); echo "prompt_ids [$tokens]"; fi
echo "$(for a in "${arms[@]}"; do printf '%s ' "$(count "$a")"; done)" >> "${TMPDIR:-/tmp}/stub-gen-loads"
[ -z "${STUB_GEN_RES:-}" ] || echo "residency lever residency=$STUB_GEN_RES why=set"
cards=''
if [ -n "${STUB_GEN_CARDS_ON:-}" ]; then
  case ${place:-a} in
    a) cards=${STUB_GEN_CARDS_A:-[NVIDIA_RTX_A6000]} ;;
    gate) cards='[NVIDIA_GeForce_RTX_3090]' ;;
    *) cards='[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090]' ;;
  esac
fi
echo "load arch=stub ctx=$ctx place=${STUB_GEN_PLACE_RAN:-${place:-a}}${cards:+ cards=$cards} (stub)"
echo "capture graph_nodes=10"
for k in "${!arms[@]}"; do
  depth=$(count "${arms[$k]}")
  if [ -n "$listed" ]; then
    echo "arm i=$k arms=${#arms[@]} ids=$depth n=$n"
    if [ -n "$sync" ]; then read -r _ || { echo "error: stdin closed before arm $k" >&2; exit 65; }; fi
    echo "prompt_ids [${arms[$k]}]"
  fi
  # STUB_COLD_GEN=<depth>:<K>: the first K arms of that depth add 100000 faults after their prompt ids.
  cm=${TMPDIR:-/tmp}/stub-cold-gen-$depth
  if [ "${STUB_COLD_GEN%%:*}" = "$depth" ] && [ "$(cat "$cm" 2> /dev/null || echo 0)" -lt "${STUB_COLD_GEN#*:}" ]; then
    echo $(($(cat "$cm" 2> /dev/null || echo 0) + 1)) > "$cm"
    sleep 0.3
    echo $(($(cat "$STUB_MAJFLT") + 100000)) > "$STUB_MAJFLT"
  fi
  mark=${TMPDIR:-/tmp}/stub-once-gen-$depth
  if [ "$depth" = "${STUB_GEN_FAIL_DEPTH:-none}" ] && [ ! -e "$mark" ]; then
    touch "$mark"
    echo "error: the stub refuses depth $depth once" >&2
    exit 3
  fi
  ns=${STUB_GEN_SLOTS_LEAK:-$(printenv BLOOMERY_GEN_SLOTS)}
  if [ "${ns:-1}" -gt 1 ] && [ -z "${STUB_GEN_SLOTS_NOPASS:-}" ]; then
    w=$((depth / ns))
    for j in $(seq 0 $((ns - 1))); do echo "step 0 $((w - 1)) 1000 slot=$j (stub)"; done
    echo "capture slots=$ns rows=1 graph_nodes=826"
    for j in $(seq 0 $((ns - 1))); do echo "time prompt n=$w ms=100.0000 tok/s=$((w * 10)).00 passes=1 kind=gemm slot=$j"; done
    for i in $(seq 1 $((n - 1))); do
      for j in $(seq 0 $((ns - 1))); do echo "step $i $((w + i - 1)) $((1000 + i)) slot=$j"; done
      echo "time pass $i ms=8.0000 positions=${STUB_GEN_SLOTS_POS:-$ns} kind=slots"
    done
    for j in $(seq 0 $((ns - 1))); do echo "tokens [$(seq -s ', ' 1000 $((1000 + n - 1)))] slot=$j"; done
    echo "SMOKE mode=graph prompt_tokens=$w depth=$w slots=$ns generated=$n warm=0 rounds=$((n - 1)) positions=$(((n - 1) * ns)) p50_ms=8.0000 mean_ms=8.0000 tok/s(aggregate)=$((ns * 125)).00 ctx=$ctx"
    continue
  fi
  echo "step 0 $depth 1000 (stub)"
  echo "time prompt n=$depth ms=100.0000 tok/s=$((depth * 10)).00 passes=1 kind=gemm"
  echo "stat prompt ubatch_tokens=0 (no ubatch ran)"
  for i in $(seq 1 $((n - 1))); do echo "step $i $((depth + i)) $((1000 + i))"; echo "time step $i ms=5.0000"; done
  echo "tokens [$(for i in $(seq 0 $((n - 1))); do [ "$i" = 0 ] && printf '%s' "${STUB_GEN_TOKEN0:-1000}" || printf ', %s' $((1000 + i)); done)]"
  if [ -n "${STUB_GEN_MTP:-}" ]; then
    # kept[k - 1] passes kept k positions: the histogram sums to the passes, its weighted sum to the positions.
    case ${STUB_GEN_PASSES:-4} in
      4) mq=4 mp=10 mk='1, 1, 1, 1' mr=200.00 ;;
      5) mq=5 mp=10 mk='2, 2, 0, 1' mr=200.00 ;;
      0) mq=0 mp=0 mk='0, 0, 0, 0' mr=0.00 ;;
      *) echo "error: the stub has no kept histogram for STUB_GEN_PASSES=$STUB_GEN_PASSES" >&2; exit 64 ;;
    esac
    echo "mtp summary proposals=$((mq > 0 ? mq - 1 : 0)) kept=[$mk] positions=$mp passes=$mq tok/s(positions)=$mr"
  fi
  [ -z "${STUB_GEN_STATS:-}" ] || echo "stat summary steps=$((n - 1)) leg_us_mean=1.0 leg_us_p50=1.0 straggle_us_max=1.0 host_slots_mean=3.4 majflt=0 minflt=0 vram_free_load=1 vram_free_min=1"
  if [ -n "${STUB_GEN_RES:-}" ]; then
    rp() { echo "residency pass pass=$1 boundary=$2 kept=$3 landed=$4 late=$5 made=$6 in_flight=0 bytes=$7 end_us=1 boundary_us=2 wait_us=0 issue_us=1 stage_us=0 prepare_us=0"; }
    rp none 0 0 0 0 1 4096
    rp prompt 1 99 1 0 1 4096
    for i in $(seq 1 $((n - 1))); do rp step $((i + 1)) 10 1 $((i == 1 ? 1 : 0)) 1 4096; done
  fi
  if [ -n "${STUB_GEN_MTP:-}" ]; then
    echo "SMOKE mode=graph prompt_tokens=$depth depth=$depth generated=$n warm=0 steps=$mp passes=$mq p50_ms=5.0000 mean_ms=5.0000 tok/s(p50)=200.00 tok/s(mean)=200.00 ctx=$ctx"
  else
    echo "SMOKE mode=graph prompt_tokens=$depth depth=$depth seeded=false generated=$n warm=0 steps=$((n - 1)) p50_ms=5.0000 mean_ms=5.0000 tok/s(p50)=200.00 ctx=$ctx"
  fi
done
EOF
cp "$T/target/release/generate_qwen3moe" "$T/base/target/release/generate_qwen3moe"
# The stub llama-server: tools/ref/stub-llama-server.py (its docstring has what it answers and the cases).
cp "$ROOT/tools/ref/stub-llama-server.py" "$T/bin/llama-server"
# The stub systemd-run: a mem= arm's scope (the runner's Mem) starts none; it appends its -p items to
# $TMPDIR/stub-systemd-run, one line a call, and runs the command after its options.
cat > "$T/bin/systemd-run" << 'EOF'
#!/usr/bin/env bash
p=()
while [ $# -gt 0 ]; do
  case $1 in
    --scope | --quiet) ;;
    -p) p+=("$2"); shift ;;
    *) break ;;
  esac
  shift
done
echo "${p[*]}" >> "${TMPDIR:-/tmp}/stub-systemd-run"
exec "$@"
EOF
chmod +x "$T/bin/"* "$T/target/release/generate_qwen3moe" "$T/base/target/release/generate_qwen3moe"
BASEBIN=$T/base/target/release/generate_qwen3moe
# The capped arms' two-shard split set (evict, evict-held): in a directory under this tree's target/, cargo's
# build directory and disk-backed on the box, not under the temporary tree: posix_fadvise(DONTNEED) evicts no
# tmpfs page, so on a tmpfs /tmp every page would stay resident and evict would be red for that reason. SH_FS
# names its filesystem, checked by the cases that read it. Each shard is 1 MiB of random bytes synced to disk,
# so its cached pages are clean; SP is a shard's pages.
mkdir -p "$ROOT/target"
SH=$(mktemp -d "$ROOT/target/depth-qwen3moe-stub-shards.XXXXXX") || exit 2
SH_FS=$(stat -f -c %T "$SH")
SHARD=$SH/q-00001-of-00002.gguf
for k in 1 2; do dd if=/dev/urandom of="$SH/q-0000$k-of-00002.gguf" bs=1M count=1 conv=fsync status=none || exit 2; done
SP=$((1048576 / $(getconf PAGESIZE)))

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
  rm -f "$tmp/tmp/stub-gen-loads" "$tmp/tmp"/stub-once-* "$tmp/tmp"/stub-cold-* "$tmp/tmp"/stub-srv-*
  rm -f "$tmp/tmp/stub-xid" "$tmp/tmp/stub-systemd-run"
  echo 0 > "$tmp/tmp/stub-majflt"
  (cd "$T" && env PATH="$T/bin:$PATH" TMPDIR="$tmp/tmp" BLOOMERY_DECODE_N=4 BLOOMERY_ARM_BOUND=60 \
    STUB_BENCH_CARDS="$T/bin/stub-bench-cards" TIMING_CARDS_POLL=1 STUB_MAJFLT="$tmp/tmp/stub-majflt" \
    BLOOMERY_CPU_BUSY_COMMS=none "${e[@]}" bash tools/ref/depth-qwen3moe.sh "$@") > "$log" 2>&1
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
  want fields "$L" 1 '^other-busy rows: 0 of 14 ' &&
  want fields "$L" 0 'ms/pass|^=== pass time|^mean pass |^ratio pass '; then
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
DRV=' \| drive_read_bytes [0-9]+ \([^)]+\)'
if want majflt "$L" 14 "$END" &&
  want majflt "$L" 14 "^ROW .*\| wall [0-9]+s $MAJ.*; ≤ [0-9.]+ % of W [0-9.]+ s\)(${DRV})?( \[cold\])?$" &&
  want majflt "$L" 2 "^ROW r[12] ours d=6 .*${MAJ}timed [0-9]+; ≤ [0-9.]+ % of W 0.1200 s\)${DRV}( \[cold\])?$" &&
  want majflt "$L" 4 "^ROW r[12] lcpp(pp)? [dp]=[46] .*${MAJ}timed [0-9]+; " &&
  want majflt "$L" 4 "^ROW r[12] mrs(pp)? [dp]=[46] .*${MAJ}timed [0-9]+; " &&
  want majflt "$L" 2 "^ROW r[12] ik d=6 .*${MAJ}whole process; ≤ [0-9.]+ % of W 0.2000 s\)" &&
  want majflt "$L" 2 "^ROW r[12] bin:base d=6 .*${MAJ}timed \? \(a one-arm run prints its prompt ids before its load: the whole process\); ≤ [0-9.]+ % of W 0.1200 s\)${DRV}( \[cold\])?$" &&
  want majflt "$L" 2 "^ROW r[12] mrs d=6 .*of W 0.0600 s\)" &&
  want majflt "$L" 4 '^ROW .*\| drive_read_bytes ' &&
  want majflt "$L" 1 '^cold rows: [0-9]+ of 14 ' &&
  want majflt "$L" 1 '^\[config\] cold tag: ' &&
  want majflt "$L" 1 '^mean ours d=6 .*\(n=2\)  \[cold [0-9]/2\] \[probe-off 0/2\]  untagged mean ([0-9.]+ tok/s \(n=[12]\)|n/a \(every row tagged\))$'; then
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

# The two-card mode (red on the runner before it).
TC=BLOOMERY_TIMING_CARDS=a6000+3090
TWO_CVD="$STUB_GPU_A6000,$STUB_GPU_3090"
L=$tmp/twocard.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "$TC" -- lcpp:6 lcpppp:4 lcppfit:6
if [ "$RC" != 0 ]; then
  fail twocard "rc $RC, want 0" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "ROW r1 lcpp d=6|ROW r1 lcpppp p=4|ROW r1 lcppfit d=6" ]; then
  fail twocard "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want twocard "$L" 1 "^ROW r1 lcpp d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 \| build stub \(0\) \| device NVIDIA RTX A6000 \(stub\) \+ NVIDIA GeForce RTX 3090 \(stub\) $END" &&
  want twocard "$L" 1 "^ROW r1 lcpppp p=4 n=0 \| tok/s\(pp\) 400.00 @ n=0, prompt 4, A6000\+3090 \| ub 512 b 2048 \(llama-bench defaults\) \| build stub \(0\) \| device NVIDIA RTX A6000 \(stub\) \+ NVIDIA GeForce RTX 3090 \(stub\) $END" &&
  want twocard "$L" 1 "^ROW r1 lcppfit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 \| fit offloaded 3/3, " &&
  want twocard "$L" 2 '^    lcpp(pp)? table \| stub k=1 ts=1.5/1.5 \| ' &&
  want twocard "$L" 1 '^    lcppfit table \| stub k= \| ' &&
  want twocard "$L" 0 '[, ]A6000 \(stub\) \|' &&
  want twocard "$L" 0 '^    timing-card: ' &&
  want twocard "$L" 8 '^    card A6000: NVIDIA RTX A6000 \(stub\), 300.00 W, 300.00 W, 2100 MHz$' &&
  want twocard "$L" 8 '^    card 3090: NVIDIA GeForce RTX 3090 \(stub\), 250.00 W, 250.00 W, 2100 MHz$' &&
  want twocard "$L" 8 '^    3090 cap: ok \(3090: NVIDIA GeForce RTX 3090 \(stub\), power.limit 250.00 W, enforced 250.00 W, bus 00000000:41:00.0 \(the 250 W cap: ok\)\)$' &&
  want twocard "$L" 8 '^    xid: 0 NVRM Xid line\(s\) since the lease was taken \(@[0-9]+\): A6000 0, 3090 0, other 0; last: none$' &&
  want twocard "$L" 1 '^\[config\] two cards: A6000\+3090, the profile.s two-card line: the stub two-card line: ' &&
  want twocard "$L" 1 "^\[config\] arms=lcpp:6 lcpppp:4 lcppfit:6 timing_gpu=$STUB_GPU_A6000 other_gpu= CUDA_VISIBLE_DEVICES=$TWO_CVD$" &&
  want twocard "$L" 1 '^\[timing-cards\] Xid count from @[0-9]+ ' &&
  want twocard "$L" 1 '^mean lcpp d=6 .*\(n=1\)'; then
  pass twocard
fi

L=$tmp/twocard-srv.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "$TC" -- lcppsrv:6 lcppsrvpp:4
if [ "$RC" != 0 ]; then
  fail twocard-srv "rc $RC, want 0" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "ROW r1 lcppsrv d=6|ROW r1 lcppsrvpp p=4" ]; then
  fail twocard-srv "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want twocard-srv "$L" 1 '^ROW r1 lcppsrv d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 \| llama-server ids=lcg: .*\| device NVIDIA RTX A6000 \(stub\) \+ NVIDIA GeForce RTX 3090 \(stub\)( |$)' &&
  want twocard-srv "$L" 1 '^ROW r1 lcppsrvpp p=4 n=0 \| tok/s\(pp\) 40.00 @ n=0, prompt 4, A6000\+3090 \| llama-server ids=lcg: .*\| device NVIDIA RTX A6000 \(stub\) \+ NVIDIA GeForce RTX 3090 \(stub\)( |$)' &&
  want twocard-srv "$tmp/tmp/stub-srv-argv" 2 ' -ts 1\.5/1\.5 '; then
  if [ -n "$(srv_left)" ]; then fail twocard-srv "servers left up: $(srv_left)" "$L"; else pass twocard-srv; fi
fi

L=$tmp/twocard-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 "$TC" -- lcpp:6 lcpppp:4
if [ "$RC" != 0 ]; then
  fail twocard-dry "rc $RC, want 0" "$L"
elif want twocard-dry "$L" 1 "^\[dry\] model=.* card=A6000\+3090 .* CUDA_VISIBLE_DEVICES=$TWO_CVD$" &&
  want twocard-dry "$L" 1 '^\[dry\] two cards: A6000\+3090, the profile.s two-card line: ' &&
  want twocard-dry "$L" 1 '^\[dry\] \[timing-cards\] A6000: NVIDIA RTX A6000 \(stub\), power.limit 300.00 W, enforced 300.00 W, bus 00000000:61:00.0$' &&
  want twocard-dry "$L" 1 '^\[dry\] \[timing-cards\] 3090: .*\(the 250 W cap: ok\)$' &&
  want twocard-dry "$L" 1 '^\[dry\] \[timing-cards\] Xid reader: journalctl -k, NVRM Xid lines on PCI:0000:61:00 \(A6000\) and PCI:0000:41:00 \(3090\)$' &&
  want twocard-dry "$L" 1 '^\[dry\] two-card precheck: ok$' &&
  want twocard-dry "$L" 1 "^\[dry\] lcpp:6: timeout --kill-after=10 60 [^ ]*/lcpp-bench -m [^ ]* -p 0 -n 4 -d 6 -r 1 -ngl 99 -fa on -ncmoe 1 -ts 1.5/1.5 --progress "; then
  pass twocard-dry
fi

# twocard_refused <name> <want rc> <pattern> <env…> -- <arms…>: the run refused with that rc, the pattern
# on one line, and no row, witness or lease.
twocard_refused() {
  local name=$1 rc=$2 pat=$3
  shift 3
  L=$tmp/$name.log
  stub_run "$L" "$@"
  if [ "$RC" != "$rc" ]; then
    fail "$name" "rc $RC, want $rc" "$L"
  elif want "$name" "$L" 1 "$pat" && want "$name" "$L" 0 '^(ROW|FAIL|DISCARD|WARMUP) |^--- witness|^\[stub\] no lease'; then
    pass "$name"
  fi
}
# The gate plan in the mode (re-pinned 2026-10-02, round armplace: twocard-arms and twocard-arms-bin held
# every ours and bin: arm refused in the mode, and an a or bp arm now runs there, twocard-place below; the
# placement the mode refuses is the gate plan, a 3090-only row in the A6000+3090 table).
twocard_refused twocard-arms 64 "^depth-qwen3moe.sh: arm '6@place=gate': place=gate is the gate plan, which loads the 3090 alone; the two-card mode times a \(plan \(a\) on the A6000, the 3090 idle\) or bp \(plan \(b′\), both cards\)$" \
  "$TC" -- lcpp:6 6@place=gate
twocard_refused twocard-arms-bin 64 "^depth-qwen3moe.sh: BLOOMERY_GEN_PLACE=gate is the gate plan, which loads the 3090 alone; " \
  "$TC" BLOOMERY_GEN_PLACE=gate -- lcpp:6 "bin:$BASEBIN:6"
twocard_refused twocard-arms-ik 64 "^depth-qwen3moe.sh: arm 'ik:6': the two-card table.s reference is mainline llama.cpp \(the profile.s two-card line: [^)]*\); ik has no two-card arm$" \
  "$TC" -- lcpp:6 ik:6
twocard_refused twocard-arms-mrs 64 "^depth-qwen3moe.sh: arm 'mrs:6': .*; mrs has no two-card arm$" \
  "$TC" -- lcpp:6 mrs:6
twocard_refused twocard-profile 64 "^depth-qwen3moe.sh: BLOOMERY_TIMING_CARDS=a6000\+3090 and the profile qwen4exp has no two-card line \(TWO_CARD_PLACEMENT\)" \
  "$TC" STUB_NO_TWO_CARD=1 -- lcpp:6
twocard_refused twocard-cap 78 "^depth-qwen3moe.sh: two cards, refused before the lease: the 3090.s power limit reads 300.00 W \(enforced 300.00 W\), not its 250 W cap" \
  "$TC" STUB_3090_LIMIT=300.00 -- lcpp:6
twocard_refused twocard-gone 69 "^depth-qwen3moe.sh: two cards, refused before the lease: the 3090 \($STUB_GPU_3090\) does not answer nvidia-smi \(rc 15\)" \
  "$TC" STUB_3090_GONE=1 -- lcpp:6
twocard_refused twocard-lease 64 "^depth-qwen3moe.sh: two cards, refused before the lease: tools/ref/lease.sh records one timing card" \
  "$TC" STUB_ONE_CARD_LEASE=1 -- lcpp:6
twocard_refused twocard-value 64 "^depth-qwen3moe.sh: BLOOMERY_TIMING_CARDS is a6000\+3090 \(the two-card mode\) or unset \(one card\), got 'both'$" \
  BLOOMERY_TIMING_CARDS=both -- lcpp:6

# A per-arm placement in the mode: a and bp arms in one run.
L=$tmp/twocard-place.log
: > "$tmp/tmp/stub-gen-place"
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" BLOOMERY_GEN_PLACE=bp STUB_GEN_CARDS_ON=1 -- 6 6@place=a lcpp:6
if [ "$RC" != 0 ]; then
  fail twocard-place "rc $RC, want 0" "$L"
elif [ "$(sort "$tmp/tmp/stub-gen-place" | paste -sd' ' -)" != "a bp" ]; then
  fail twocard-place "the processes' --place: $(paste -sd' ' - < "$tmp/tmp/stub-gen-place"), want a and bp (a load each)" "$L"
elif want twocard-place "$L" 1 '^ROW r1 ours d=6 n=4 ctx=256 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place bp \| ' &&
  want twocard-place "$L" 1 '^ROW r1 ours@place=a d=6 n=4 ctx=256 \| tok/s\(mean\) [0-9.]+ @ n=4, depth 6, A6000\+3090 \| place a \| ' &&
  want twocard-place "$L" 1 '^ROW r1 lcpp d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000\+3090 ' &&
  want twocard-place "$L" 1 '^ratio d=6 +ours/ours@place=a +mean ' &&
  want twocard-place "$L" 1 '^\[config\] placements: 6 bp, 6@place=a a \(an arm.s place= over BLOOMERY_GEN_PLACE=bp\)$' &&
  want twocard-place "$L" 0 '^FAIL '; then
  pass twocard-place
fi
L=$tmp/twocard-place-3090.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" STUB_GEN_CARDS_ON=1 \
  'STUB_GEN_CARDS_A=[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090]' -- 6@place=a
if [ "$RC" != 1 ]; then
  fail twocard-place-3090 "rc $RC, want 1" "$L"
elif want twocard-place-3090 "$L" 1 "^FAIL r1 ours@place=a d=6 rc=0 \| two cards: the engine.s load record names cards \[NVIDIA_RTX_A6000,NVIDIA_GeForce_RTX_3090\], and --place a loads the A6000 alone: the 3090 must stay idle; last line: " &&
  want twocard-place-3090 "$L" 0 '^ROW '; then
  pass twocard-place-3090
fi
L=$tmp/twocard-nocards.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 "$TC" -- 6
if [ "$RC" != 1 ]; then
  fail twocard-nocards "rc $RC, want 1" "$L"
elif want twocard-nocards "$L" 1 "^FAIL r1 ours d=6 rc=0 \| two cards: the engine.s load record names no cards: which cards it loaded cannot be told; last line: " &&
  want twocard-nocards "$L" 0 '^ROW '; then
  pass twocard-nocards
fi
L=$tmp/twocard-place-dry.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1 "$TC" BLOOMERY_GEN_PLACE=bp -- 6 6@place=a
if [ "$RC" != 0 ]; then
  fail twocard-place-dry "rc $RC, want 0" "$L"
elif want twocard-place-dry "$L" 1 '^\[dry\] 6@place=a: one arm of a load: .* --ctx 256 --place a --time --arm-sync   # load key [^ ]*\|ctx=256\|place=a$' &&
  want twocard-place-dry "$L" 1 '^\[dry\] 6: one arm of a load: .* --ctx 256 --place bp --time --arm-sync   # load key [^ ]*\|ctx=256\|place=bp$' &&
  want twocard-place-dry "$L" 1 '^\[dry\] placements: 6 bp, 6@place=a a \(an arm.s place= over BLOOMERY_GEN_PLACE=bp\)$' &&
  want twocard-place-dry "$L" 0 'env place='; then
  pass twocard-place-dry
fi
# One card.
twocard_refused bp-onecard-arm 64 "^depth-qwen3moe.sh: arm '6@place=bp': place=bp is plan \(b′\), which loads on both cards \(the A6000 and its 3090 expert tier\); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000\+3090$" \
  -- 6 6@place=bp
twocard_refused place-ref 64 "^depth-qwen3moe.sh: arm 'lcpp:6@place=a': place= sets generate_qwen3moe's --place, and lcpp is a reference engine's arm, which takes no placement of ours$" \
  -- 6 lcpp:6@place=a
twocard_refused place-word 64 "^depth-qwen3moe.sh: arm '6@place=b2': place=b2: generate_qwen3moe takes --place a, gate or bp in this runner$" \
  -- 6 6@place=b2
twocard_refused place-twice 64 "^depth-qwen3moe.sh: arm '6@place=a,place=a': place is given twice$" \
  -- 6 6@place=a,place=a
twocard_refused place-card-arm 64 "^depth-qwen3moe.sh: arm '6@place=gate': place=gate is the gate plan, which loads on the 3090, and the timing card is " \
  -- 6 6@place=gate
L=$tmp/place-one.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_AB_WARMUP=0 -- 6 6@place=a
if [ "$RC" != 0 ]; then
  fail place-one "rc $RC, want 0" "$L"
elif want place-one "$L" 1 '^ROW r1 ours d=6 ' &&
  want place-one "$L" 1 '^ROW r1 ours@place=a d=6 .* \| place a \| ' &&
  want place-one "$L" 0 '^FAIL '; then
  pass place-one
fi

L=$tmp/twocard-xid.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "$TC" STUB_BENCH_XID=pp4 -- lcpp:6 lcpppp:4 lcpp:5
if [ "$RC" != 1 ]; then
  fail twocard-xid "rc $RC, want 1" "$L"
elif [ "$(heads "$L" | paste -sd'|' -)" != "ROW r1 lcpp d=6|FAIL r1 lcpppp p=4|ROW r1 lcpp d=5" ]; then
  fail twocard-xid "the rows' heads: $(heads "$L" | paste -sd'|' -)" "$L"
elif want twocard-xid "$L" 1 "^FAIL r1 lcpppp p=4 rc=0 \| two cards: 1 NVRM Xid line\(s\) since the last arm.s check \(A6000 0, 3090 1 since the lease was taken\); last: [0-9.]+ ws kernel: NVRM: Xid \(PCI:0000:41:00\): 79, .*GPU has fallen off the bus\.; last line: " &&
  want twocard-xid "$L" 4 '^    xid: 1 NVRM Xid line\(s\) since the lease was taken \(@[0-9]+\): A6000 0, 3090 1, other 0; last: ' &&
  want twocard-xid "$L" 1 '^    dropped: lcpppp at 4$' &&
  want twocard-xid "$L" 1 '^failed arms: r1 lcpppp p=4 rc=0; $'; then
  pass twocard-xid
fi

L=$tmp/twocard-one.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "$TC" 'STUB_SEE_ONE=tg4 @ d6' -- lcpp:6 lcpppp:4
if [ "$RC" != 1 ]; then
  fail twocard-one "rc $RC, want 1" "$L"
elif want twocard-one "$L" 1 "^FAIL r1 lcpp d=6 rc=0 \| two cards: the engine saw 1 CUDA device\(s\) \(device 0 'NVIDIA RTX A6000 \(stub\)', device 1 '\?'\), not the A6000 and the 3090" &&
  want twocard-one "$L" 1 '^ROW r1 lcpppp p=4 '; then
  pass twocard-one
fi

L=$tmp/twocard-busy.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 "$TC" STUB_BUSY_ONCE=3090 -- lcpp:6
if [ "$RC" != 0 ]; then
  fail twocard-busy "rc $RC, want 0" "$L"
elif want twocard-busy "$L" 1 "^\[cards-busy\] .* compute apps on a timed card: \[$STUB_GPU_3090, 4242, 100 MiB;\]; waiting up to 10 min$" &&
  want twocard-busy "$L" 1 '^\[cards-busy\] .* both cards are free after 1 s$' &&
  want twocard-busy "$L" 1 '^--- witness wait-cards ' &&
  want twocard-busy "$L" 1 '^ROW r1 lcpp d=6 '; then
  pass twocard-busy
fi

L=$tmp/twocard-optin.log
# shellcheck disable=SC2016 # the inner shell expands them
(cd "$T" && env PATH="$T/bin:$PATH" "$TC" bash -c 'source tools/ref/timing-card.sh; echo "TIMING_GPU=[$TIMING_GPU] CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES] TIMING_CARDS=[$TIMING_CARDS]"' other-runner.sh) > "$L" 2>&1
if want twocard-optin "$L" 1 '^\[timing-cards\] refused: BLOOMERY_TIMING_CARDS=a6000\+3090, and other-runner.sh has no two-card mode \(a runner opts in with TIMING_CARDS_RUNNER=1\): it would time the A6000 alone$' &&
  want twocard-optin "$L" 1 '^TIMING_GPU=\[\] CUDA_VISIBLE_DEVICES=\[\] TIMING_CARDS=\[\]$'; then
  pass twocard-optin
fi
L=$tmp/srv.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 -- 6 lcppsrv:6 lcppsrvpp:4 lcppsrvpp8:4 lcppsrvfit:6
TAIL='\| wall [0-9]+s \| majflt 0 \(timed 0; ≤ 0.0 % of W'
if [ "$RC" != 0 ]; then
  fail srv "rc $RC, want 0" "$L"
elif [ -n "$(srv_left)" ]; then
  fail srv "servers still up: $(srv_left)" "$L"
elif want srv "$L" 1 "^ROW r1 lcppsrv d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) \| llama-server ids=lcg: warm-up 20.00 tok/s majflt 0, continuation same \| prompt_n 6 prompt tok/s 60.00 \| build 1 \(stub\) \| device Stub Card $TAIL 0.2500 s\)$" &&
  want srv "$L" 1 "^ROW r1 lcppsrvpp p=4 n=0 \| tok/s\(pp\) 40.00 @ n=0, prompt 4, A6000 \(stub\) \| llama-server ids=lcg: warm-up 40.00 tok/s\(pp\) majflt 0, continuation same \| ub 512 b 2048 \(llama-server defaults\) \| build 1 \(stub\) \| device Stub Card $TAIL 0.1000 s\)$" &&
  want srv "$L" 1 '^ROW r1 lcppsrvpp8 p=4 .*\| ub 8 b 2048 \(the arm.s lever\) \|' &&
  want srv "$L" 1 '^ROW r1 lcppsrvfit d=6 n=4 \| tok/s 20.00 @ n=4, depth 6, A6000 \(stub\) \| fit offloaded 3/3, overridden CPU:2 in blk 1-1 \(blk 1: 2\), buffers CUDA0 10.00 CPU_Mapped 2.00 MiB \| llama-server ids=lcg: ' &&
  want srv "$L" 1 '^ratio d=6 +ours/lcppsrv ' &&
  want srv "$L" 1 '^ratio d=6 +ours/lcppsrvfit ' &&
  want "srv argv" "$tmp/tmp/stub-srv-argv" 2 '^-m [^ ]+ -ngl 99 -fa on -ncmoe 1 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 256 --host 127.0.0.1 --port 0$' &&
  want "srv argv" "$tmp/tmp/stub-srv-argv" 1 '^-m [^ ]+ -fa on -fit on -fitt 1024 -v -np 1 '; then
  pass srv
fi

L=$tmp/srv-exit.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_SRV_EXIT=1 -- lcppsrv:6
if [ "$RC" != 1 ]; then
  fail srv-exit "rc $RC, want 1" "$L"
elif want srv-exit "$L" 1 "^FAIL r1 lcppsrv d=6 rc=5 \| -c 256 \(ours' --ctx at that D or P: D \+ N rounded up to 256\): llama-server exited 5 before it answered /health" &&
  want srv-exit "$L" 0 '^ROW '; then
  pass srv-exit
fi

L=$tmp/srv-cold.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_WARM_ROWS=1 STUB_SRV_COLD=2 -- lcppsrv:6
if [ "$RC" != 0 ]; then
  fail srv-cold "rc $RC, want 0" "$L"
elif want srv-cold "$L" 1 '^COLD r1 lcppsrv d=6 .*\| wall [0-9]+s \| majflt 1000000 \(timed 1000000; ≤ [0-9.]+ % of W 0.2500 s\) \[cold\]$' &&
  want srv-cold "$L" 1 '^ROW r1 lcppsrv d=6 .* continuation same \(the retry\) .*\(timed 0; ' &&
  want srv-cold "$L" 1 '^warm rows: 1 COLD row\(s\) ran once more: 1 clean on the retry, 0 FAIL rc=cold$'; then
  pass srv-cold
fi

mkdir -p "$T/data/qwen4exp"
seq 100 800 > "$T/data/qwen4exp/corpus-prose.ids"
prose512=$(seq 100 611 | paste -sd, -)
L=$tmp/srv-prose.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DATA="$T/data" -- prose:512 lcppsrv:prose:512
if [ "$RC" != 0 ]; then
  fail srv-prose "rc $RC, want 0" "$L"
elif [ -n "$(srv_left)" ]; then
  fail srv-prose "servers still up: $(srv_left)" "$L"
elif want srv-prose "$L" 1 '^ROW r1 lcppsrv@prose d=512 n=4 \| tok/s 20.00 @ n=4, depth 512, A6000 \(stub\) \| llama-server ids=prose: ' &&
  want srv-prose "$L" 1 '^ratio prose d=512 +ours@prose/lcppsrv@prose ' &&
  want srv-prose "$L" 0 '^ratio d=512 ' &&
  want srv-prose "$L" 1 '^ratio pp prose p=512 +ours@prose/lcppsrv@prose ' &&
  want srv-prose "$L" 1 '^xcheck prose p=512 r1 ours@prose\(r1\)/lcppsrv@prose: same 4 \(1000,1001,1002,1003\)$' &&
  want srv-prose "$L" 1 '^cpu-busy rows: 0 of 2 ' &&
  want "srv-prose ids" "$tmp/tmp/stub-srv-reqs" 2 "^[12] 512 4 $prose512$" &&
  want "srv-prose argv" "$tmp/tmp/stub-srv-argv" 1 ' -c 768 --host 127\.0\.0\.1 --port 0$'; then
  pass srv-prose
fi
L=$tmp/srv-xcheck.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DATA="$T/data" STUB_GEN_TOKEN0=7 -- prose:512 lcppsrv:prose:512
if [ "$RC" != 1 ]; then
  fail srv-xcheck "rc $RC, want 1" "$L"
elif want srv-xcheck "$L" 1 '^FAIL xcheck prose p=512 r1 ours@prose\(r1\)/lcppsrv@prose: differs at 0 of 4 \| ours@prose 7,1001,1002,1003 \| lcppsrv@prose 1000,1001,1002,1003$' &&
  want srv-xcheck "$L" 1 '^failed arms: r1 xcheck lcppsrv@prose p=512 \(differs at 0 of 4\); $' &&
  want srv-xcheck "$L" 1 '^    dropped: lcppsrv@prose at 512$' &&
  want srv-xcheck "$L" 0 '^ratio prose d=512 +ours@prose/lcppsrv@prose '; then
  L=$tmp/srv-xcheck-tail.log
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_DATA="$T/data" STUB_SRV_TOKEN1=7 -- prose:512 lcppsrv:prose:512
  if [ "$RC" != 0 ]; then
    fail srv-xcheck "tail: rc $RC, want 0" "$L"
  elif want srv-xcheck "$L" 1 '^xcheck prose p=512 r1 ours@prose\(r1\)/lcppsrv@prose: first id same, differs at 1 of 4 \| ours@prose 1000,1001,1002,1003 \| lcppsrv@prose 1000,7,1002,1003 \[xcheck-tail\]$' &&
    want srv-xcheck "$L" 1 '^xcheck: 1 server row\(s\): 0 same, 1 \[xcheck-tail\] ' &&
    want srv-xcheck "$L" 1 '^ratio prose d=512 +ours@prose/lcppsrv@prose '; then
    pass srv-xcheck
  fi
fi

L=$tmp/place-gate.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_GEN_PLACE=gate BLOOMERY_TIMING_GPU="$STUB_GPU_3090" -- 6
if [ "$RC" != 0 ]; then
  fail place-gate "rc $RC, want 0" "$L"
elif want place-gate "$L" 1 '^ROW r1 ours d=6 n=4 ctx=256 \| tok/s\(mean\) 200.00 @ n=4, depth 6, [^|]* \| place gate \| ' &&
  want "place-gate argv" "$tmp/tmp/stub-gen-place" 1 '^gate$'; then
  pass place-gate
fi
L=$tmp/place-ran.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_GEN_PLACE=gate BLOOMERY_TIMING_GPU="$STUB_GPU_3090" STUB_GEN_PLACE_RAN=a -- 6
if [ "$RC" != 1 ]; then
  fail place-ran "rc $RC, want 1" "$L"
elif want place-ran "$L" 1 '^FAIL r1 ours d=6 rc=0 \| its load line names place=a; the runner passed --place gate'; then
  pass place-ran
fi
L=$tmp/place-card.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_GEN_PLACE=gate BLOOMERY_TIMING_GPU="$STUB_GPU_A6000" -- 6
if [ "$RC" != 64 ]; then
  fail place-card "rc $RC, want 64" "$L"
# Re-pinned 2026-10-02 (round armplace): the refusal is tools/ref/arm-place.sh's one text in every runner,
# which names the plan before the card; the case still pins the gate word, the 3090 and the timing card.
elif want place-card "$L" 1 '^depth-qwen3moe.sh: BLOOMERY_GEN_PLACE=gate is the gate plan, which loads on the 3090, and the timing card is '; then
  pass place-card
fi

# Two copies of generate_qwen3moe's schema: without the residency kinds, and with generate_ds41's rows of
# them, so the case reads the same whether or not this tree's schema declares them.
QS=$T/tools/bloomery/schema/generate_qwen3moe.jsonl
cp "$QS" "$tmp/qschema.keep"
qschema() { # qschema <file> <with 0|1>: the kept schema less its residency rows, plus generate_ds41's when 1
  python3 - "$tmp/qschema.keep" "$T/tools/bloomery/schema/generate_ds41.jsonl" "$2" > "$1" << 'PY'
import json, sys
keep, ds41, add = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
res = ("residency_lever", "residency_host", "residency_pass", "residency_reset")
lines = open(keep).read().splitlines()
head, rows = json.loads(lines[0]), [l for l in lines[1:] if json.loads(l).get("kind") not in res]
if add:
    rows += [l for l in open(ds41).read().splitlines()[1:] if json.loads(l).get("kind") in res]
head["kinds"] = len(rows)
print(json.dumps(head, separators=(",", ":")))
print("\n".join(rows))
PY
}
qschema "$QS" 0
L=$tmp/res-noschema.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_RES=mid-p148-s1 -- 6@BLOOMERY_RESIDENCY=mid-p148-s1
RC1=$RC
qschema "$QS" 1
L2=$tmp/res-sums.log
stub_run "$L2" BLOOMERY_AB_ROUNDS=1 STUB_GEN_RES=mid-p148-s1 STUB_GEN_STATS=1 -- 6@BLOOMERY_RESIDENCY=mid-p148-s1
RC2=$RC
L3=$tmp/res-nolever.log
stub_run "$L3" BLOOMERY_AB_ROUNDS=1 -- 6@BLOOMERY_RESIDENCY=mid-p148-s1
RC3=$RC RC=$RC2
cp "$tmp/qschema.keep" "$QS"
if [ "$RC1" != 1 ]; then
  fail res-sums "a schema with no residency kind: rc $RC1, want 1" "$L"
elif [ "$RC" != 0 ]; then
  fail res-sums "rc $RC, want 0" "$L2"
elif want res-sums "$L" 1 "^FAIL r1 ours@[^ ]* d=6 rc=0 \| the arm runs BLOOMERY_RESIDENCY=mid-p148-s1, and generate_qwen3moe's checked-in schema \(tools/bloomery/schema/generate_qwen3moe.jsonl\) declares no residency record, " &&
  want res-sums "$L2" 1 '^ROW r1 ours@[^ ]* d=6 .* \| host slots/token 3\.4 \| residency mid-p148-s1 \(set\) passes 3 kept 30 landed 3 late 1 made 3 bytes 12288 \| ' &&
  want res-sums "$L2" 1 '^residency mean ours@[^ ]* d=6 mid-p148-s1: rows 1, passes/row 3\.0, kept/pass 10\.00, landed/pass 1\.000, late/pass 0\.333, made/pass 1\.000$'; then
  pass res-sums
fi
if [ "$RC3" != 1 ]; then
  fail res-nolever "an arm that asks for residency and prints no lever record: rc $RC3, want 1" "$L3"
elif want res-nolever "$L3" 1 "^FAIL r1 ours@[^ ]* d=6 rc=0 \| the arm runs BLOOMERY_RESIDENCY=mid-p148-s1 and printed no residency lever record, which its schema declares"; then
  pass res-nolever
fi

L=$tmp/cpu-guard.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_CPU_BUSY=1 -- 6 lcpp:6
if [ "$RC" != 0 ]; then
  fail cpu-guard "rc $RC, want 0" "$L"
elif want cpu-guard "$L" 1 '^ROW r1 ours d=6 .*\[cpu-busy\]' &&
  want cpu-guard "$L" 1 '^ROW r1 lcpp d=6 .*\[cpu-busy\]' &&
  want cpu-guard "$L" 1 '^cpu-busy rows: 2 of 2 '; then
  pass cpu-guard
fi

# The real qwen4exp profile's 3090 line: -ncmoe 43 when the timing card is the 3090, 26 otherwise, and a
# timing card that is not the A6000 with no 3090 UUID resolved refused by name.
mkdir -p "$T/tools/ref/models-real"
cp "${DEPTH_QWEN3MOE_PROFILE:-$ROOT/tools/ref/models/qwen4exp.sh}" "$T/tools/ref/models-real/qwen4exp.sh"
# shellcheck disable=SC2016 # the profile's names expand in the child shell
prof() { (cd "$T" && env -i PATH="$PATH" BLOOMERY_TIMING_GPU="$1" bash -c 'source tools/ref/models-real/qwen4exp.sh && echo "flags=$LCPP_GPU_FLAGS"' 2>&1); }
p_a=$(prof "$STUB_GPU_A6000") p_3=$(prof "$STUB_GPU_3090") p_0=$(prof '')
cp "$T/tools/ref/cards.sh" "$tmp/cards.sh.keep"
echo "GPU_A6000=$STUB_GPU_A6000" > "$T/tools/ref/cards.sh"
p_x=$(prof GPU-99999999)
rc_x=$?
cp "$tmp/cards.sh.keep" "$T/tools/ref/cards.sh"
case "$p_a|$p_3|$p_0" in
  *'-ncmoe 26 '*'|'*'-ncmoe 43 '*'|'*'-ncmoe 26 '*)
    if [ "$rc_x" = 64 ] && [[ $p_x == *'sizes QWEN38_NCMOE (43 vs 26)'* ]]; then pass profile-3090; else fail profile-3090 "no 3090 UUID: rc $rc_x, $p_x"; fi
    ;;
  *) fail profile-3090 "A6000: $p_a; 3090: $p_3; unset: $p_0" ;;
esac

L=$tmp/mtp.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_MTP=1 -- 6
if [ "$RC" != 0 ]; then
  fail mtp "rc $RC, want 0" "$L"
elif want mtp "$L" 1 '^ROW r1 ours d=6 .* \| mtp E\(4\) 2\.500 = positions 10 / passes 4, kept \[1, 1, 1, 1\] \| ' &&
  want mtp "$L" 1 '^ROW r1 ours d=6 .* \| ms/pass 12\.5000 \(mean_ms × steps 10 / passes 4\) \| '; then
  pass mtp
fi

# Two MTP arms whose passes differ, their tok/s alike: ours keeps 10 positions in 4 passes, the lever arm in 5,
# both at mean_ms 5, so ms/pass is 5 × 10 / 4 = 12.5 and 5 × 10 / 5 = 10.0. The ratio is over passes a second,
# 1000 / 12.5 = 80 and 1000 / 10 = 100, ours over the arm: 80 / 100 = 0.8000 in each round (above 1 would be
# ours faster, as in the decode table, whose ratio for the pair is 200 / 200 = 1.0000).
L=$tmp/mtp-pass-ratio.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 STUB_GEN_MTP=1 -- 6 6@STUB_GEN_PASSES=5
if [ "$RC" != 0 ]; then
  fail mtp-pass-ratio "rc $RC, want 0" "$L"
elif want mtp-pass-ratio "$L" 2 '^ROW r[12] ours d=6 .* \| ms/pass 12\.5000 \(mean_ms × steps 10 / passes 4\) \| ' &&
  want mtp-pass-ratio "$L" 2 '^ROW r[12] ours@STUB_GEN_PASSES=5 d=6 .* \| mtp E\(4\) 2\.000 = positions 10 / passes 5, kept \[2, 2, 0, 1\] \| ms/pass 10\.0000 \(mean_ms × steps 10 / passes 5\) \| ' &&
  want mtp-pass-ratio "$L" 1 '^=== pass time per arm ' &&
  want mtp-pass-ratio "$L" 2 '^mean pass ' &&
  want mtp-pass-ratio "$L" 1 '^mean pass ours d=6 +12\.5000 ms/pass  \[12\.5000\.\.12\.5000\]  \(n=2\)  \[cold [0-2]/2\] \[probe-off 0/2\]  untagged mean (12\.5000 ms/pass \(n=[12]\)|n/a \(every row tagged\))$' &&
  want mtp-pass-ratio "$L" 1 '^mean pass ours@STUB_GEN_PASSES=5 d=6 +10\.0000 ms/pass  \[10\.0000\.\.10\.0000\]  \(n=2\)  \[cold [0-2]/2\] \[probe-off 0/2\]  untagged mean (10\.0000 ms/pass \(n=[12]\)|n/a \(every row tagged\))$' &&
  want mtp-pass-ratio "$L" 1 '^ratio pass ' &&
  want mtp-pass-ratio "$L" 1 '^ratio pass d=6 +ours/ours@STUB_GEN_PASSES=5 +mean 0\.8000 ± 0\.0000 \(n=2\)  of means 0\.8000  per round: r1 0\.8000 r2 0\.8000  cpu-busy: ours 0/2, ours@STUB_GEN_PASSES=5 0/2  cold: ours [0-2]/2, ours@STUB_GEN_PASSES=5 [0-2]/2  probe-off: ours 0/2$' &&
  want mtp-pass-ratio "$L" 1 '^ratio d=6 +ours/ours@STUB_GEN_PASSES=5 +mean 1\.0000 '; then
  pass mtp-pass-ratio
fi

L=$tmp/mtp-pass-bad.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_MTP=1 STUB_GEN_PASSES=0 -- 6
if [ "$RC" != 1 ]; then
  fail mtp-pass-bad "rc $RC, want 1" "$L"
elif want mtp-pass-bad "$L" 1 "^FAIL r1 ours d=6 rc=0 \| its SMOKE line carries passes= and no pass time can be read from it \(passes='0' steps='0'; " &&
  want mtp-pass-bad "$L" 0 '^ROW ' &&
  want mtp-pass-bad "$L" 0 '^mean pass ' &&
  want mtp-pass-bad "$L" 1 '^failed arms: r1 ours d=6 rc=0; $'; then
  pass mtp-pass-bad
fi

# The aggregate arm: the plain arm at 5 ms a step (1000 / 5 = 200 tok/s) and the 2-slot arm at 8 ms a round
# of 2 positions over its 3 rounds (2 · 3 · 1000 / 24 = 250 tok/s): 250 / 200 = 1.2500 in each round.
L=$tmp/slots.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 -- 6 6@BLOOMERY_GEN_SLOTS=2
if [ "$RC" != 0 ]; then
  fail slots "rc $RC, want 0" "$L"
elif ! grep -qx '12 ' "$tmp/tmp/stub-gen-loads"; then
  fail slots "no process fed the aggregate arm's 12 ids: $(paste -sd'|' - < "$tmp/tmp/stub-gen-loads")" "$L"
elif want slots "$L" 2 '^ROW r[12] ours@BLOOMERY_GEN_SLOTS=2 d=6 n=4 ctx=256 \| tok/s\(aggregate\) 250\.00 @ n=2·3, depth 6, [^|]* \| slots 2 \| tok/s\(per stream, mean\) 125\.00 \| p50 8\.0000 ms/pass \| mean 8\.0000 ms/pass \| .* \| nodes 826 \| ' &&
  want slots "$L" 1 '^mean ours@BLOOMERY_GEN_SLOTS=2 d=6 +250\.00 tok/s  \[250\.00\.\.250\.00, spread 0\.00%\]  \(aggregate of 2 slots\) \(n=2\)' &&
  want slots "$L" 1 '^mean ours d=6 +200\.00 tok/s ' &&
  want slots "$L" 1 '^ratio slots ' &&
  want slots "$L" 1 '^ratio slots d=6 +ours@BLOOMERY_GEN_SLOTS=2/ours +mean 1\.2500 ± 0\.0000 \(n=2\)  of means 1\.2500  per round: r1 1\.2500 r2 1\.2500 ' &&
  want slots "$L" 0 '^ratio d=6 .*GEN_SLOTS'; then
  pass slots
fi
L=$tmp/slots-pos.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_SLOTS_POS=1 -- 6@BLOOMERY_GEN_SLOTS=2
if [ "$RC" != 1 ]; then
  fail slots-pos "rc $RC, want 1" "$L"
elif want slots-pos "$L" 1 '^FAIL r1 ours@BLOOMERY_GEN_SLOTS=2 d=6 rc=0 \| 3 of its 3 counted kind=slots records hold positions other than its 2 slots' &&
  want slots-pos "$L" 0 '^ROW '; then
  pass slots-pos
fi
L=$tmp/slots-leak.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_SLOTS_LEAK=2 -- 6
if [ "$RC" != 1 ]; then
  fail slots-leak "rc $RC, want 1" "$L"
elif want slots-leak "$L" 1 '^FAIL r1 ours d=6 rc=0 \| its output holds 3 time pass record\(s\) of kind=slots and the arm names no BLOOMERY_GEN_SLOTS' &&
  want slots-leak "$L" 0 '^ROW '; then
  pass slots-leak
fi
L=$tmp/slots-none.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_SLOTS_NOPASS=1 -- 6@BLOOMERY_GEN_SLOTS=2
if [ "$RC" != 1 ]; then
  fail slots-none "rc $RC, want 1" "$L"
elif want slots-none "$L" 1 '^FAIL r1 ours@BLOOMERY_GEN_SLOTS=2 d=6 rc=0 \| the arm runs BLOOMERY_GEN_SLOTS=2 and printed no counted time pass record of kind=slots' &&
  want slots-none "$L" 0 '^ROW '; then
  pass slots-none
fi
# Two builds' aggregate arms on one prompt and list: the bin: arm's one process and ours' load each fed the
# corpus's first 2 · 3 ids, both at 2 positions a round over 3 rounds at 8 ms (250 tok/s): 250 / 250 a round.
L=$tmp/bin-slots.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 BLOOMERY_DATA="$T/data" -- "bin:$BASEBIN:prose:3@BLOOMERY_GEN_SLOTS=2" prose:3@BLOOMERY_GEN_SLOTS=2
if [ "$RC" != 0 ]; then
  fail bin-slots "rc $RC, want 0" "$L"
elif [ "$(paste -sd'|' - < "$tmp/tmp/stub-gen-loads")" != "6 |6 |6 |6 " ]; then
  fail bin-slots "the processes' id counts: $(paste -sd'|' - < "$tmp/tmp/stub-gen-loads"), want 6 |6 |6 |6 " "$L"
elif want bin-slots "$L" 0 '^FAIL ' &&
  want bin-slots "$L" 2 '^ROW r[12] bin:base@prose@BLOOMERY_GEN_SLOTS=2 d=3 n=4 ctx=256 \| tok/s\(aggregate\) 250\.00 @ n=2·3, depth 3, [^|]* \| slots 2 \| ' &&
  want bin-slots "$L" 2 '^ROW r[12] ours@prose@BLOOMERY_GEN_SLOTS=2 d=3 n=4 ctx=256 \| tok/s\(aggregate\) 250\.00 @ n=2·3, depth 3, [^|]* \| slots 2 \| ' &&
  want bin-slots "$L" 1 '^mean bin:base@prose@BLOOMERY_GEN_SLOTS=2 d=3 +250\.00 tok/s  \[250\.00\.\.250\.00, spread 0\.00%\]  \(aggregate of 2 slots\) \(n=2\)' &&
  want bin-slots "$L" 1 '^ratio slots: bin:base@prose@BLOOMERY_GEN_SLOTS=2 has no plain twin bin:base@prose among the arms: no ratio$' &&
  want bin-slots "$L" 1 '^ratio slots bin ' &&
  want bin-slots "$L" 1 '^ratio slots bin d=3 +ours@prose@BLOOMERY_GEN_SLOTS=2/bin:base@prose@BLOOMERY_GEN_SLOTS=2 +mean 1\.0000 ± 0\.0000 \(n=2\)  of means 1\.0000  per round: r1 1\.0000 r2 1\.0000 ' &&
  want bin-slots "$L" 0 '^ratio (d|prose d)='; then
  pass bin-slots
fi

# The capped arms (the runner's Mem). A bin: arm's mem= in the scope an ours arm's takes: the stub systemd-run's
# -p items once a capped process, the scope's witness, the evict line over the profile's (empty) file, the row's
# capped field; the plain arm beside it unscoped.
L=$tmp/bin-mem.log
stub_run "$L" BLOOMERY_AB_ROUNDS=2 -- "bin:$BASEBIN:6@mem=29G" 6
if [ "$RC" != 0 ]; then
  fail bin-mem "rc $RC, want 0" "$L"
elif [ "$(paste -sd'|' - < "$tmp/tmp/stub-systemd-run")" != "MemoryMax=29G MemorySwapMax=0|MemoryMax=29G MemorySwapMax=0" ]; then
  fail bin-mem "the stub systemd-run's calls: $(paste -sd'|' - < "$tmp/tmp/stub-systemd-run"), want two of MemoryMax=29G MemorySwapMax=0" "$L"
elif want bin-mem "$L" 2 '^ROW r[12] bin:base@mem=29G d=6 n=4 ctx=256 \| tok/s\(mean\) 200\.00 @ n=4, depth 6, [^|]* \| capped mem=29G \| p50 ' &&
  want bin-mem "$L" 2 '^ROW r[12] ours d=6 ' &&
  want bin-mem "$L" 0 '^ROW r[12] ours d=6 .*capped' &&
  want bin-mem "$L" 2 '^    mem scope: MemoryMax=29G MemorySwapMax=0 peak=' &&
  want bin-mem "$L" 2 '^    evict: [^ ]*/model-00001-of-00001\.gguf shards=1 pages=0 resident before=0 after=0 ' &&
  want bin-mem "$L" 1 '^\[config\] mem: bin:[^ ]*@mem=29G 29G — ' &&
  want bin-mem "$L" 0 '^FAIL '; then
  pass bin-mem
fi
# evict: every capped process's eviction over the two-shard set on disk, each process (STUB_GEN_READ_MODEL=1)
# reading the shards back in, so every eviction finds all 2·SP pages resident and leaves none.
L=$tmp/evict.log
cat "$SH"/q-*-of-00002.gguf > /dev/null
stub_run "$L" BLOOMERY_AB_ROUNDS=2 STUB_GEN_READ_MODEL=1 -- "bin:$BASEBIN:6@mem=29G,model=$SHARD" "6@mem=29G,model=$SHARD"
if [ "$SH_FS" = tmpfs ] || [ "$SH_FS" = ramfs ]; then
  fail evict "the shard directory $SH is on $SH_FS, whose pages posix_fadvise does not evict" "$L"
elif [ "$RC" != 0 ]; then
  fail evict "rc $RC, want 0" "$L"
elif want evict "$L" 4 '^    evict: ' &&
  want evict "$L" 4 "^    evict: $SHARD shards=2 pages=$((2 * SP)) resident before=$((2 * SP)) after=0 \(posix_fadvise DONTNEED, then mincore\)$" &&
  want evict "$L" 2 '^ROW r[12] bin:base@mem=29G,model=[^ ]* d=6 .* \| capped mem=29G \| ' &&
  want evict "$L" 2 '^ROW r[12] ours@mem=29G,model=[^ ]* d=6 .* \| capped mem=29G \| ' &&
  want evict "$L" 4 '^    mem scope: MemoryMax=29G '; then
  pass evict
fi
# evict-held: a holder process maps the first shard and touches every page, so posix_fadvise leaves its SP pages
# resident: the capped arm a FAIL row naming the count, its process never started, rc 1.
L=$tmp/evict-held.log
rm -f "$tmp/holder.ready"
python3 -c 'import mmap, sys, time
f = open(sys.argv[1], "rb")
m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
s = sum(m[i] for i in range(0, len(m), mmap.PAGESIZE))
open(sys.argv[2], "w").write("held %d\n" % s)
time.sleep(600)' "$SHARD" "$tmp/holder.ready" &
echo "$!" > "$tmp/holder.pid"
hp=$(cat "$tmp/holder.pid") w=0
while [ ! -s "$tmp/holder.ready" ] && kill -0 "$hp" 2> /dev/null && [ "$w" -lt 100 ]; do
  sleep 0.1
  w=$((w + 1))
done
if [ "$SH_FS" = tmpfs ] || [ "$SH_FS" = ramfs ]; then
  fail evict-held "the shard directory $SH is on $SH_FS, whose pages posix_fadvise does not evict"
elif [ ! -s "$tmp/holder.ready" ]; then
  fail evict-held "the holder never mapped $SHARD (no ready file after $w polls)"
else
  stub_run "$L" BLOOMERY_AB_ROUNDS=1 STUB_GEN_READ_MODEL=1 -- "6@mem=29G,model=$SHARD"
  if [ "$RC" != 1 ]; then
    fail evict-held "rc $RC, want 1" "$L"
  elif [ -s "$tmp/tmp/stub-gen-loads" ]; then
    fail evict-held "the capped arm's process started: $(paste -sd'|' - < "$tmp/tmp/stub-gen-loads")" "$L"
  elif want evict-held "$L" 1 "^    evict: $SHARD shards=2 pages=$((2 * SP)) resident before=[0-9]+ after=$SP " &&
    want evict-held "$L" 1 "^FAIL r1 ours@mem=29G,model=[^ ]* d=6 rc=evict \| evict: $SP of the $((2 * SP)) pages of $SHARD's 2 shard\(s\) stay resident after posix_fadvise\(DONTNEED\) " &&
    want evict-held "$L" 0 '^ROW ' &&
    want evict-held "$L" 1 '^failed arms: r1 ours@mem=29G,model=[^ ]* d=6 rc=evict; $'; then
    pass evict-held
  fi
fi
kill "$hp" 2> /dev/null
wait "$hp" 2> /dev/null
rm -f "$tmp/holder.pid"

L=$tmp/warm.log
stub_run "$L" BLOOMERY_AB_ROUNDS=1 BLOOMERY_WARM_ROWS=1 STUB_COLD_GEN=4:2 'STUB_COLD_BENCH=tg4 @ d6:2' -- 6 4 lcpp:6
want_seq="PRIME r1 ours d=6
ROW r1 ours d=6
PRIME r1 ours d=4
COLD r1 ours d=4
PRIME r1 ours d=4
ROW r1 ours d=4
COLD r1 lcpp d=6
FAIL r1 lcpp d=6"
seq=$(grep -oE '^(DISCARD|WARMUP|PRIME|COLD|ROW|FAIL) r[0-9]+ [^ ]+ [dp]=[0-9]+' "$L")
if [ "$RC" != 1 ]; then
  fail warm "rc $RC, want 1" "$L"
elif [ "$seq" != "$want_seq" ]; then
  fail warm "the rows' heads: $(echo "$seq" | paste -sd'|' -)" "$L"
elif [ "$(paste -sd'|' - < "$tmp/tmp/stub-gen-loads")" != "6 6 4 4 |4 4 " ]; then
  fail warm "the processes' --arm lists: $(paste -sd'|' - < "$tmp/tmp/stub-gen-loads"), want 6 6 4 4 |4 4 " "$L"
elif want warm "$L" 1 '^FAIL r1 lcpp d=6 rc=cold \| cold after warm-up and one retry \(timed 100000; ' &&
  want warm "$L" 1 '^    dropped: lcpp at 6$' &&
  want warm "$L" 1 '^mean ours d=4 .*\(n=1\)' &&
  want warm "$L" 1 '^warm rows: 2 COLD row\(s\) ran once more: 1 clean on the retry, 1 FAIL rc=cold$'; then
  pass warm
fi

if [ "${DEPTH_QWEN3MOE_STUB_SHOW:-}" = 1 ]; then
  for L in "$tmp"/rotate.log "$tmp"/rotate-dry.log "$tmp"/mrs-noiter.log "$tmp"/warmup-rotate.log \
    "$tmp"/blocks.log "$tmp"/order-bad.log "$tmp"/warmup-bad.log "$tmp"/blocks-dry.log "$tmp"/ref-fail.log \
    "$tmp"/discard-fail.log "$tmp"/discard-nofit.log "$tmp"/warmup-fail.log "$tmp"/group-fail.log "$tmp"/twocard*.log \
    "$tmp"/srv*.log "$tmp"/place-*.log "$tmp"/mtp*.log "$tmp"/slots*.log "$tmp"/bin-slots.log "$tmp"/bin-mem.log \
    "$tmp"/evict*.log "$tmp"/cpu-guard.log "$tmp"/warm.log; do
    echo "--- ${L##*/}"
    cat "$L"
  done
fi
echo "depth-qwen3moe-stub: $n checks, $failed failed"
[ "$failed" = 0 ]
