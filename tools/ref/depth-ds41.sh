#!/usr/bin/env bash
# V4.1 decode by depth and prefill by prompt length, three engines in one lease on the timing card
# (run on the box, lead-only): our engine (generate_ds41 --depth D --time, placement (a)), ik
# (llama-bench -gp D,N, or -p P -n 0 for prefill, at the profile's IK_GPU_FLAGS, under its
# IK_GPU_ENV) and mainline llama.cpp (llama-bench -d D, or -p P -n 0, at the profile's
# LCPP_GPU_FLAGS), alternated arm by arm.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/depth-ds41.sh 6 ik:6 lcpp:6'
#   just depth-gpu-ds41 6 lcpp:6 4096 lcpp:4096
#   just depth-gpu-ds41 512 ikpp:512 lcpppp:512 4096 ikpp:4096 lcpppp:4096
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-ds41 6 lcpp:6    # the command lines, no lease, no load
#
# The V4.1 sibling of depth-gpu.sh, and it blocks the same failure: a ratio read at one depth and
# quoted as "decode is faster" — a step's attention term grows with the cached keys, so the depth
# goes into every number (`tok/s @ n=N, depth D, <card>`).
#
# Arms, in lease order (the order rotates by one slot each round — the position bias ab-decode.sh
# names):
#   <D>      ours: generate_ds41 --depth D -n N --time. The fed ids are lease.sh's lcg_prompt D,
#            one real decode step each, untimed; generated token 0 comes out of the last of them and
#            the N - 1 steps after it are timed. The context is the binary's default, the serving
#            ctx_max the plan (a) is made for, at every depth: the plan's card expert prefix depends
#            on ctx_max, so a per-depth ctx would move experts between the card and the host.
#            The row also carries `pp_tok/s <v> (n=D, passes=K)` from the binary's `time prompt`
#            row, the wall of those D fed steps (Prefill below): one arm reports both the prefill
#            of D ids and the decode at depth D.
#   ik:<D>   ik: llama-bench -p 0 -n 0 -gp D,N -r 1 (D = 0: plain tg N). llama-bench sizes its
#            context to D + N itself and feeds its own prompt ids, not lcg_prompt's. It runs as
#            `env $IK_GPU_ENV`: without GGML_CUDA_NO_PINNED_WEIGHTS the CPU expert overrides turn
#            the host context into one pinned allocation larger than RAM and the load fails (the
#            profile's IK_GPU_ENV comment).
#   lcpp:<D> mainline: llama-bench -p 0 -n N -d D -r 1 $LCPP_GPU_FLAGS (D = 0: plain tg N).
#            Mainline has no -gp; -d prefills D tokens before its clock starts, and its row label
#            is `tgN @ dD` where ik's is `tgN@ppD`. llama-bench sizes n_ctx to D + N and the
#            context pads it to 256 (src/llama-context.cpp). No environment: mainline keeps the
#            host context on the file mapping (the profile's LCPP_GPU_FLAGS comment).
#   lcpp<K>:<D>  the same with --n-cpu-moe K in place of the profile's LCPP_NCMOE: the sweep arm.
#            The profile's LCPP_NCMOE_SWEEP names the values that load, e.g.
#            `lcpp:6 lcpp34:6 lcpp35:6` interleaved with `6`.
#   ikpp:<P> ik's prefill: llama-bench -p P -n 0 -r 2 -o json with the ik:<D> arm's binary, flags
#            ($IK_GPU_FLAGS) and environment ($IK_GPU_ENV); the row is the second repetition,
#            P / samples_ns[1], as `tok/s(pp) … @ n=0, prompt P`, and names the first as `cold`.
#            The file is larger than RAM, so an arm that follows another engine refaults its own
#            pages during its first repetition; one load, two samples, the warm one counts.
#   lcpppp:<P>  mainline's prefill, the same at $LCPP_GPU_FLAGS, the lcpp:<D> arm's.
#   ikpp<U>:<P>, lcpppp<U>:<P>  the same with -ub U -b max(U, 2048): the ubatch lever, not a default
#            (Prefill below). Refused when the profile's flags already name -ub or -b.
#   <D>@NAME=VALUE[,NAME=VALUE...]  ours at depth D with those variables set (`env NAME=VALUE ...`):
#            a lever arm of the same binary, row label `ours@NAME=VALUE[,...]`. Beside a plain `<D>`
#            arm it is the same-binary A/B, e.g. `6 6@BLOOMERY_LAUNCH_THREAD=1`.
#            An arm with BLOOMERY_DRAFT=dspark also sees the other card, where the draft runs, and
#            gets the profile's DSPARK_MODEL unless it names one (timing-card.sh dspark_env).
#   prose:<P>[@NAME=VALUE[,NAME=VALUE...]]  ours fed the first P ids of
#            $BLOOMERY_DATA/engram/corpus-prose.ids (one id per line) instead of the LCG prompt:
#            generate_ds41 --tokens <those ids> -n N --time, with the variables set as in <D>@…. Row label
#            `prose`, or `prose@NAME=VALUE[,...]`; the depth column is P. The prompt's routing, and so
#            its card and host work, is prose's, not the LCG walk's: a prose arm is compared only with
#            prose arms of the same P — `prose:512 prose:512@BLOOMERY_CARD_EXPERTS=expert` is the
#            same-binary A/B on the prose prompt — in its own decode and prefill tables (prose / each
#            prose@ label), never with ours or the references. A P the file cannot supply (P < 1, or
#            past its line count) is refused before anything runs.
#   bin:<path>:<D>  a second generate_ds41 (an absolute path on the box, a base tree's build) at depth
#            D, row label `bin:<basename of its tree>` (the tree is the path above `target/`). It is a
#            base by construction, so its freshness is not asked; its tree line (sha256, HEAD, dirty
#            files) is printed with the references'.
# Every label is its own engine in the per-arm means and in the ratio table (ours / each other label,
# the prose labels excepted; prose / each prose@ label in the prose tables).
# Prefill values (ours' `time prompt`, the pp arms) have their own means and ratio table per prompt
# length P, never the decode ones'.
# Our arms run at any depth up to the plan's ctx_max (every indexer layer selects its list at every
# position); generate_ds41 refuses only D + N - 1 > --ctx, and a refusal stops the runner (rc 1).
#
# Placement. Ours is plan (a): every layer and the head on the A6000, each routed layer's experts
# [0, n_l) on the card (n_l 63-64 of 384, the budget's), the rest on the host tier — the plan line
# generate_ds41 prints. ik moves experts by tensor, and a layer's 384 experts are one tensor, so it
# cannot keep a prefix of every layer; the profile's --n-cpu-moe keeps the experts of the first
# layers on the CPU and the last ones' whole on the card, sized to the bytes plan (a) gives the
# card's experts. The per-token host work is the same in expectation (6 routed experts per layer,
# each with the host's share of the probability), the shape is not: ours joins the host once per
# layer, ik's card layers never wait on the host and its CPU layers never use the card's share.
# Mainline places by the same rule (--n-cpu-moe), so an lcpp row has ik's shape; the profile sizes
# both counts per file from MODEL (IK_NCMOE and LCPP_NCMOE, one arithmetic — the two engines place
# the same tensors on the card), so the two references hold the same layers on the card.
#
# Paging. The two engines' host expert sets differ (ours: experts n_l.. of every layer, ik: all
# experts of its CPU layers), and together they are about the page cache's size. The witness
# prints the major fault count and the page cache before and after every arm, so an arm that paged
# its set back in shows. BLOOMERY_GEN_WARM (generate_ds41 --warm) trims our arm's first steps.
# With BLOOMERY_STEP_STATS=1 (BLOOMERY_BOX_ENV on the Mac side) our arm's `stat summary` line is
# echoed with its load lines; the per-step `stat step` lines stay in the arm's output only.
#
# Prefill. pp_tok/s is P over the wall of processing a P-token prompt, in each engine's terms:
#   ours     generate_ds41's `time prompt` row: before the first fed step to after the readback of
#            generated token 0, `passes` the steps it took. V4.1's ours is one decode step per
#            token today, so its pp equals its step rate at the fed depths by construction — at or
#            above it: the plain feed is one body per id and one readback at the end, a timed step
#            one body and one readback. A DSpark arm's feed reads back per id and also reads each
#            position's features into the draft (`kind=dspark`).
#   ik, lcpp llama-bench's pp test: llama_decode over the P ids in batches of -b and ubatches of
#            -ub, then one synchronize (test_prompt), one repetition, llama-bench's own value.
#            Both trees default to -ub 512 -b 2048; the row names the batch sizes that ran.
# The arms do not start alike. The warmup before llama-bench's timed repetition is a 1-token prompt
# in ik (examples/llama-bench/llama-bench.cpp:2230) and the whole prompt in mainline
# (tools/llama-bench/llama-bench.cpp:2378 on the V4.1 branch); ours has none: the prompt is the
# process's first work after the capture. So ik's and our rows carry first-touch costs mainline's
# does not; the witness's page cache and pgmajfault lines show the paging part of them.
# Where the host experts run in prefill: at -ub 512 ik's CUDA backend copies no host MUL_MAT_ID
# weight to the card (it does at ubatch × 6 used experts >= 32 × 384, -ub >= 2048 [derived, from the
# profile's IK_NCMOE comment and ggml/src/ggml-cuda.cu:5226 at the default offload batch size]), and
# mainline runs -nopo 1, so both prefill the host layers' experts on the host threads, as ours does
# at every position. The ubatch lever:
# ikpp<U> at U >= 2048 crosses ik's offload line, whose compute buffer holds a layer's expert
# tensors, more than the profile's placement leaves free on the card [derived, the profile's
# LCPP_GPU_FLAGS comment] — pair it with an IK_GPU_FLAGS override of --n-cpu-moe inside the box
# command; lcpppp<U> keeps -nopo 1.
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 3),
# BLOOMERY_GEN_WARM, BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_ARM_BOUND
# (seconds one arm may run, default 900: a hung arm fails the runner with rc 124/137 instead of
# holding the lease), BLOOMERY_AB_WARMUP (below), BLOOMERY_DRY=1 (print each arm's command line, the
# binaries' tree lines, the CPU guard's settings and reading, the warm-up and the rotation, then exit
# 0 before the lease: nothing is loaded and nothing is timed).
#
# Warm-up. The lease's first process reads the model's pages cold — the plan's reads of the file and
# the host tier's experts through the mapping, both inside our arm's timed window — so the first row
# of a lease reads low. The runner runs the first arm once before round 1 and discards it: its row
# prints as `WARMUP r0 …`, between its own witness blocks and after the same guards, and is in no
# mean, ratio or row count; a warm-up that fails stops the runner as a round's arm does. It costs one
# row's wall, 73-128 s for the V4.1 rows at P = 512 and 4096 (lcg and prose) [derived: those rows'
# `wall` column], so the lease is that much longer. BLOOMERY_AB_WARMUP=0 skips it; any value but 0 or
# 1 is refused. depth-qwen3moe.sh has none: its model sits on the card, so no timed window reads the
# file.
#
# Contention. Before every arm the runner checks both the other card (guard_other, timing-card.sh:
# `[other-busy]`) and the CPU (guard_cpu, lease.sh: `[cpu-busy]` when builds or reference engines it
# did not start — BLOOMERY_CPU_BUSY_COMMS — sum past BLOOMERY_CPU_BUSY_PCT percent of one cpu), and
# checks the CPU again after the arm; a row that met CPU contention ends in ` [cpu-busy]`, one whose
# arm started beside a compute process on the other card in ` [other-busy]` (guard_other's
# OTHER_BUSY_TAG), and the closing summary counts both. BLOOMERY_OTHER_STRICT=1 aborts (rc 75) on
# either instead.
set -uo pipefail
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_ENV, LCPP, LCPPBIN, LCPP_GPU_FLAGS,
# LCPP_NCMOE); tools/box.sh exports its MODEL to our binary as BLOOMERY_REF_MODEL, so the three
# engines open one file.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = deepseek41 ] || {
  echo "depth-ds41.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
N=${BLOOMERY_DECODE_N:-96}
WARM=${BLOOMERY_GEN_WARM:-}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_ds41}
BOUND=${BLOOMERY_ARM_BOUND:-900}
DRY=${BLOOMERY_DRY:-}
AB_WARMUP=${BLOOMERY_AB_WARMUP:-1}
case $AB_WARMUP in
  0 | 1) ;;
  *) echo "depth-ds41.sh: BLOOMERY_AB_WARMUP is 1 (the default: one discarded run of the first arm) or 0, got '$AB_WARMUP'" >&2; exit 64 ;;
esac
# The word an arm's row starts with: ROW, a row of the tables, or WARMUP, the discarded warm-up run.
# counted: whether the row goes into the sums and the row counts (only a ROW row does).
ROW_TAG=ROW
counted() { [ "$ROW_TAG" = ROW ]; }
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6)
ours=0 ik=0 lcpp=0
# Per arm, by its index in ARMS: the kind (ours, prose, ref or bin), the depth, the row label, the
# reference engine (ref), the binary (ours and bin) and the NAME=VALUE list (comma-separated).
A_KIND=() A_DEP=() A_LABEL=() A_ENG=() A_BIN=() A_ENV=()
# A prose arm's prompt, the ids as generate_ds41 --tokens takes them (empty for every other arm).
A_TOK=()
# The prose prompts' file and its id count, read once when an arm names it.
PROSE=${BLOOMERY_DATA:-}/engram/corpus-prose.ids PROSE_N=
# A prefill arm's engine (ikpp[<U>], lcpppp[<U>]), and its ubatch lever U (empty: the default).
pp_eng() { case $1 in ikpp* | lcpppp*) return 0 ;; *) return 1 ;; esac; }
pp_ub() { local u=${1#ikpp}; echo "${u#lcpppp}"; }
arm_usage() {
  echo "depth-ds41.sh: arm '$1' is <D>, <D>@NAME=VALUE[,NAME=VALUE...], prose:<P>[@NAME=VALUE,...], ik:<D>, lcpp:<D>, lcpp<K>:<D>, ikpp[<U>]:<P>, lcpppp[<U>]:<P> or bin:<path>:<D>" >&2
  exit 64
}
# arm_envs_ok <arm> <NAME=VALUE list>: the list is one or more NAME=VALUE, no spaces or commas in a value.
arm_envs_ok() {
  local -a kv
  IFS=, read -r -a kv <<< "$2"
  [ ${#kv[@]} -gt 0 ] || arm_usage "$1"
  for e in "${kv[@]}"; do
    [[ $e =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:],]+$ ]] || arm_usage "$1"
  done
}
# prose_check <arm> <P>: PROSE_N read once; P outside 1..PROSE_N is refused.
prose_check() {
  if [ -z "$PROSE_N" ]; then
    [ -r "$PROSE" ] || { echo "depth-ds41.sh: arm '$1': no prose prompt file at $PROSE (BLOOMERY_DATA)" >&2; exit 2; }
    PROSE_N=$(($(wc -l < "$PROSE")))
  fi
  if [ "$2" -lt 1 ] || [ "$2" -gt "$PROSE_N" ]; then
    echo "depth-ds41.sh: arm '$1': a prose prompt of $2 ids; $PROSE holds $PROSE_N (1..$PROSE_N)" >&2
    exit 64
  fi
}
for a in "${ARMS[@]}"; do
  kind=ref eng=${a%%:*} dep=${a#*:} label='' bin='' envs='' tok=''
  case $a in
    prose:*)
      kind=prose eng=prose dep=${a#prose:} bin=$BIN label=prose
      case $dep in *@*) envs=${dep#*@} dep=${dep%%@*} label=prose@$envs && arm_envs_ok "$a" "$envs" ;; esac
      case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
      prose_check "$a" "$dep"
      tok=$(head -n "$dep" "$PROSE" | paste -sd, -)
      ours=1
      ;;
    bin:*)
      kind=bin eng=bin bin=${a#bin:}
      dep=${bin##*:} bin=${bin%:*}
      case $bin in /*) ;; *) arm_usage "$a" ;; esac
      tree=${bin%/target/*}
      [ "$tree" != "$bin" ] || tree=${bin%/*}
      label=bin:${tree##*/}
      ;;
    *@*)
      kind=ours eng=ours dep=${a%%@*} envs=${a#*@} bin=$BIN label=ours@$envs
      arm_envs_ok "$a" "$envs"
      ours=1
      ;;
    *:*)
      case $eng in
        ik) ik=1 ;;
        lcpp | lcpp[0-9] | lcpp[0-9][0-9]) lcpp=1 ;;
        ikpp | ikpp[1-9]*) ik=1 ;;
        lcpppp | lcpppp[1-9]*) lcpp=1 ;;
        *) arm_usage "$a" ;;
      esac
      label=$eng
      ;;
    *) kind=ours eng=ours dep=$a bin=$BIN label=ours ours=1 ;;
  esac
  case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
  if pp_eng "$eng"; then
    ub=$(pp_ub "$eng")
    case $ub in *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-ds41.sh: arm '$a': a prompt of 0 ids has no prefill to time" >&2; exit 64; }
    if [ -n "$ub" ]; then
      case $eng in ikpp*) flags=$IK_GPU_FLAGS ;; *) flags=$LCPP_GPU_FLAGS ;; esac
      case " $flags " in
        *" -ub "* | *" --ubatch-size "* | *" -b "* | *" --batch-size "*)
          echo "depth-ds41.sh: arm '$a': the profile's flags already set the batch sizes ($flags); the ubatch lever would add a second value" >&2
          exit 64
          ;;
      esac
    fi
  fi
  A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=("$bin") A_ENV+=("$envs") A_TOK+=("$tok")
done
# The card pin, the card's witness lines, the other-card guard and the binary's freshness.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# The t quantiles of the ratio intervals, df 1..ROUNDS: tools/ref/tdist.py, the table gpu-ab.py and
# card.py read.
T975=$(python3 "${BASH_SOURCE[0]%/*}/tdist.py" "$ROUNDS") || {
  rc=$?
  echo "depth-ds41.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS (rc $rc; ROUNDS is a positive integer)" >&2
  exit "$rc"
}
# Each engine's binary matters only to its own arms: a reference-only run neither builds ours (the
# recipe skips the build) nor reads it, so its freshness is not asked. A dry run asks nothing of
# our binary: it prints the command line it would run.
if [ "$ours" = 1 ] && [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
# A bin:<path> arm's binary is checked where its tree line is taken, below.
if [ "$ik" = 1 ]; then [ -x "$IKBIN" ] || { echo "depth-ds41.sh: no llama-bench at $IKBIN" >&2; exit 2; }; fi
if [ "$lcpp" = 1 ]; then [ -x "$LCPPBIN" ] || { echo "depth-ds41.sh: no llama-bench at $LCPPBIN" >&2; exit 2; }; fi
CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
# A binary's sha256 and its tree's HEAD and dirty count, once. GIT_OPTIONAL_LOCKS=0 keeps `git
# status` from rewriting the index of a tree this root process does not own. A tree box.sh synced
# has no .git (the rsync leaves it out): when that tree is the one this runner stands in, its commit
# is box.sh's BLOOMERY_GIT_COMMIT (`-dirty` when the Mac tree held uncommitted changes), printed as
# `head=<commit>(box.sh) dirty_files=?`; any other tree without git prints `head=? dirty_files=?`.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" | cut -c1-12)
  if head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=9 HEAD 2> /dev/null); then
    dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  else
    head='?' dirty='?'
    if [ -n "${BLOOMERY_GIT_COMMIT:-}" ] && [ "$(cd "$tree" 2> /dev/null && pwd -P)" = "$(pwd -P)" ]; then
      head="$BLOOMERY_GIT_COMMIT(box.sh)"
    fi
  fi
  echo "$bin sha256=$sha head=$head dirty_files=$dirty"
}
IK_LINE='' LCPP_LINE='' BIN_LINES=()
[ "$ik" = 0 ] || IK_LINE=$(tree_line "$IKBIN" "$IK")
# shellcheck disable=SC2153 # LCPP is the profile's, which shellcheck does not follow
[ "$lcpp" = 0 ] || LCPP_LINE=$(tree_line "$LCPPBIN" "$LCPP")
for i in "${!ARMS[@]}"; do
  [ "${A_KIND[$i]}" = bin ] || continue
  b=${A_BIN[$i]}
  [ -x "$b" ] || { echo "depth-ds41.sh: arm '${ARMS[$i]}': no binary at $b" >&2; exit 2; }
  t=${b%/target/*}
  [ "$t" != "$b" ] || t=${b%/*}
  BIN_LINES+=("${A_LABEL[$i]}: $(tree_line "$b" "$t")")
done
ref_witness() {
  [ -z "$IK_LINE" ] || echo "    ik: $IK_LINE"
  [ -z "$LCPP_LINE" ] || echo "    lcpp: $LCPP_LINE"
  [ ${#BIN_LINES[@]} -eq 0 ] || printf '    %s\n' "${BIN_LINES[@]}"
}

# The witness before and after every row: the timing card's lines (with our binary), the busiest
# processes, the page cache and the major fault count.
WITNESS=(head-open indent card busiest model mem pgmajfault)

# ref_cmd <engine> <depth or prompt length>: the reference arm's binary, arguments and row label,
# into REF_ENV, REF_BIN, REF_ARGS and REF_LABEL, and for a prefill arm the batch sizes its row names
# into REF_BATCH. The flags are word-split on purpose: the profile keeps them as one string. An
# lcpp<K> engine is LCPP_GPU_FLAGS with its --n-cpu-moe value replaced by K. A prefill arm is its
# decode twin's binary, flags and environment with -p P -n 0 in place of the decode test.
ref_cmd() {
  local eng=$1 dep=$2 flags ub reps=(-r 1)
  REF_ENV=() REF_BATCH=
  case $eng in
    ik | ikpp*)
      # shellcheck disable=SC2206
      REF_ENV=($IK_GPU_ENV)
      REF_BIN=$IKBIN flags=$IK_GPU_FLAGS
      ;;
    *)
      REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS
      case $eng in
        lcpp[0-9]*)
          flags=$(echo " $flags " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
          flags="${flags# }--n-cpu-moe ${eng#lcpp}"
          ;;
      esac
      ;;
  esac
  case $eng in
    ikpp* | lcpppp*)
      ub=$(pp_ub "$eng")
      REF_ARGS=(-p "$dep" -n 0) REF_LABEL="pp$dep |" REF_BATCH="ub 512 b 2048 (llama-bench defaults)"
      reps=(-r 2 -o json)
      if [ -n "$ub" ]; then
        REF_ARGS+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))")
        REF_BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
      fi
      ;;
    ik) if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n 0 -gp "$dep,$N"); REF_LABEL="tg$N@pp$dep |"; fi ;;
    *) if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n "$N" -d "$dep"); REF_LABEL="tg$N @ d$dep |"; fi ;;
  esac
  # shellcheck disable=SC2206
  REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" "${reps[@]}" $flags)
}

# One reference arm: its llama-bench on its flags, the row, and the sum.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 raw rc val build dev t0 t1 fail cold errf
  ref_cmd "$eng" "$dep"
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  if pp_eng "$eng"; then
    # The json goes to stdout alone; the loader's log goes to a file, shown on a failure.
    errf=${TMPDIR:-/tmp}/depth-ds41-$eng-d$dep-r$r.err
    raw=$(timeout --kill-after=10 "$BOUND" env "${REF_ENV[@]}" "$REF_BIN" "${REF_ARGS[@]}" 2>"$errf")
  else
    raw=$(timeout --kill-after=10 "$BOUND" env "${REF_ENV[@]}" "$REF_BIN" "${REF_ARGS[@]}" 2>&1)
  fi
  rc=$?
  t1=$(date +%s)
  witness "post r$r $eng d=$dep"
  guard_cpu "post r$r $eng d=$dep"
  if pp_eng "$eng"; then
    # Exactly one test and two samples, or no value: a missing field is a failed arm, not a 0.
    val=$(echo "$raw" | jq -r 'if length == 1 and (.[0].samples_ns | length) == 2
      then .[0] | (.n_prompt * 1e9 / .samples_ns[1] * 100 | round / 100) else empty end' 2>/dev/null)
    cold=$(echo "$raw" | jq -r '.[0] | (.n_prompt * 1e9 / .samples_ns[0] * 100 | round / 100)' 2>/dev/null)
    [ -z "$val" ] && raw="$raw
$(tail -n 40 "$errf" 2>/dev/null)"
  else
    val=$(echo "$raw" | grep -F "$REF_LABEL" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
  fi
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    # The whole output goes to a file: the loader's reason for a failed load is many lines
    # above the tail.
    fail=${TMPDIR:-/tmp}/depth-ds41-$eng-d$dep-r$r.log
    echo "$raw" > "$fail"
    echo "r$r $eng d=$dep produced no '${REF_LABEL% |}' row (rc $rc); full output: $fail" >&2
    echo "$raw" | tail -n 20 >&2
    exit 1
  fi
  # The reference's own table, header and row: its columns name every setting it ran with that
  # differs from its defaults, so the log shows which flags took.
  if pp_eng "$eng"; then
    # The json's settings, samples dropped: the same "which flags took" as the md table.
    echo "$raw" | jq -c '.[0] | del(.samples_ns, .samples_ts)' | sed "s/^/    $eng params /"
    build=$(echo "$raw" | jq -r '.[0] | "\(.build_commit) (\(.build_number))"')
    dev=$(echo "$raw" | jq -r '.[0].gpu_info')
  else
    echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
    build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
    dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
  fi
  if pp_eng "$eng"; then
    echo "$ROW_TAG r$r $eng p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME | cold ${cold:-?} (repetition 1 of 2) | $REF_BATCH | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$CPU_BUSY_TAG$OTHER_BUSY_TAG"
    counted || return 0
    pp_sums+=("$eng|$dep|$r|$val|$CPU_BUSY_TAG$OTHER_BUSY_TAG")
  else
    echo "$ROW_TAG r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$CPU_BUSY_TAG$OTHER_BUSY_TAG"
    counted || return 0
    sums+=("$eng|$dep|$r|$val|")
  fi
  count_row
}

# The closing summary's row counts: every ROW line, and those that carried each contention tag.
count_row() {
  n_rows=$((n_rows + 1))
  [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
  [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
}

# parse_pp: the `time prompt n=<P> ms=<ms> tok/s=<v> passes=<K> kind=<k>` row of a generate_ds41
# run on stdin, as `<P> <v> <K> <k>`; nothing when the run printed none (a base tree's build).
parse_pp() {
  sed -nE 's/^time prompt n=([0-9]+) ms=[0-9.]+ tok\/s=([0-9.]+|inf) passes=([0-9]+) kind=([a-z]+)$/\1 \2 \3 \4/p' | head -n 1
}

# pp_col <arm kind> <what> <output>: an ours or bin arm's prefill column from its run's output, into
# PP_COL, with the parsed P and tok/s in PP_N and PP_TPS for the closing tables. A bin arm's base
# build may print no time prompt row (PP_N empty, the column says so); an ours arm's binary is this
# tree's, so a missing row stops the runner.
pp_col() {
  local passes kind
  read -r PP_N PP_TPS passes kind <<< "$(echo "$3" | parse_pp)"
  if [ -n "$PP_N" ]; then
    PP_COL=" | pp_tok/s $PP_TPS (n=$PP_N, passes=$passes)"
    [ "$kind" = steps ] || PP_COL="${PP_COL%)}, kind=$kind)"
  elif [ "$1" = bin ]; then
    PP_COL=" | pp_tok/s ? (the binary prints no time prompt row)"
  else
    echo "$2 produced no time prompt row" >&2
    echo "$3" | tail -n 20 >&2
    exit 1
  fi
}

# The variables arm <i> of ours runs with, into ARM_ENVS: its own, and for a DSpark arm
# (BLOOMERY_DRAFT=dspark) the other card's visibility and the draft file (timing-card.sh dspark_env).
# The dry run prints the same list.
arm_envs() {
  ARM_ENVS=()
  [ -z "${A_ENV[$1]}" ] || IFS=, read -r -a ARM_ENVS <<< "${A_ENV[$1]}"
  if [[ ",${A_ENV[$1]}," == *",BLOOMERY_DRAFT=dspark,"* ]]; then
    local -a extra=()
    mapfile -t extra < <(dspark_env)
    ARM_ENVS=("${extra[@]}" "${ARM_ENVS[@]}")
  fi
}
# The prompt arm <i> of ours feeds, into ARM_FEED: --tokens <the prose ids> for a prose arm, else
# --depth <D> (the binary's LCG prompt).
arm_feed() {
  if [ "${A_KIND[$1]}" = prose ]; then ARM_FEED=(--tokens "${A_TOK[$1]}"); else ARM_FEED=(--depth "${A_DEP[$1]}"); fi
}
# One arm of a generate_ds41: ours (our binary, with the arm's variables when it has any) or a
# second binary. The row and the sum under the arm's label.
# ours_arm <index> <round>
ours_arm() {
  local i=$1 r=$2 dep label bin out rc t0 t1 smoke p50 mean warmcol series draft h10 t10 uniq_tok tps_mean tps_p50
  local -a envs=() feed=()
  dep=${A_DEP[$i]} label=${A_LABEL[$i]} bin=${A_BIN[$i]}
  arm_envs "$i"
  envs=("${ARM_ENVS[@]}")
  arm_feed "$i"
  feed=("${ARM_FEED[@]}")
  witness "pre r$r $label d=$dep n=$N"
  t0=$(date +%s)
  if [ ${#envs[@]} -eq 0 ]; then
    out=$(timeout --kill-after=10 "$BOUND" "$bin" "${feed[@]}" -n "$N" --time ${WARM:+--warm "$WARM"} 2>&1)
  else
    out=$(timeout --kill-after=10 "$BOUND" env "${envs[@]}" "$bin" "${feed[@]}" -n "$N" --time ${WARM:+--warm "$WARM"} 2>&1)
  fi
  rc=$?
  t1=$(date +%s)
  witness "post r$r $label d=$dep n=$N"
  guard_cpu "post r$r $label d=$dep"
  if [ $rc -ne 0 ]; then
    echo "r$r $label d=$dep FAILED rc=$rc" >&2
    echo "$out" | tail -n 20 >&2
    exit 1
  fi
  smoke=$(echo "$out" | grep -E '^SMOKE ')
  [ -n "$smoke" ] || { echo "r$r $label d=$dep produced no SMOKE line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
  echo "$out" | grep -E '^(plan|load|capture|fed|prefill|stat prefill|stat summary|time prompt) '
  pp_col "${A_KIND[$i]}" "r$r $label d=$dep" "$out"
  # shellcheck disable=SC2001 # a regex capture, which ${var//} does not have
  p50=$(echo "$smoke" | sed 's/.*p50_ms=\([0-9.]*\).*/\1/')
  # shellcheck disable=SC2001 # the same
  mean=$(echo "$smoke" | sed 's/.*mean_ms=\([0-9.]*\).*/\1/')
  warmcol=$(echo "$smoke" | sed -n 's/.*warm=\([0-9]*\).*/\1/p')
  # `time step` rows are one position each; under BLOOMERY_DRAFT the rows are `time pass … positions=1|2`
  # and the `draft summary` line carries the positions-per-second rate the verdict reads.
  series=$(echo "$out" | awk '/^time (step|pass) /{sub(/.*ms=/,""); sub(/ .*/,""); print}')
  draft=$(echo "$out" | grep -E '^draft summary ' | sed -n 's/.*proposals=\([0-9]*\) accepts=\([0-9]*\) positions=\([0-9]*\) passes=\([0-9]*\) tok\/s(positions)=\([0-9.]*\).*/p=\1\/\4 q=\2\/\1 positions=\3 tok\/s(positions)=\5/p')
  h10=$(echo "$series" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  t10=$(echo "$series" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  uniq_tok=$(echo "$out" | awk '/^step / && $2 != 0 {print $4}' | sort -u | wc -l | tr -d ' ')
  tps_mean=$(awk -v m="$mean" 'BEGIN{printf "%.2f", 1e3/m}')
  tps_p50=$(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}')
  echo "$ROW_TAG r$r $label d=$dep n=$N | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok${draft:+ | draft $draft}$PP_COL | wall $((t1 - t0))s$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  counted || return 0
  count_row
  if [ -n "$draft" ]; then tps_mean=${draft##*tok/s(positions)=}; fi
  sums+=("$label|$dep|$r|$tps_mean|$tps_p50")
  [ -z "$PP_N" ] || pp_sums+=("$label|$PP_N|$r|$PP_TPS|$CPU_BUSY_TAG$OTHER_BUSY_TAG")
}

# ratio_table <prefix> <keys> <labels> <tagged> [base]: records `label|key|round|value|extra` on stdin;
# for every key and every label of <labels>, each round's base / label ratio (arms that ran more than
# once in a round averaged first), their mean with its 95 % interval (Student t at rounds - 1 degrees
# of freedom, T975) and the ratio of the arm means. The base is ours (the default), or prose for the
# prose tables.
# With tagged = 1 the extra field is the row's contention tags, and the line ends with each side's
# count of them.
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v tagged="$4" -v base="${5:-ours}" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
  if (tagged) { if ($5 ~ /cpu-busy/) bc[a]++; if ($5 ~ /other-busy/) bo[a]++ }
} END {
  nt = split(t975, t, " ")
  nd = split(deps, d, " ")
  nr = split(refs, rf, " ")
  for (i = 1; i <= nd; i++) for (j = 1; j <= nr; j++) {
    ref = rf[j]
    if (!((base SUBSEP d[i]) in an) || !((ref SUBSEP d[i]) in an)) continue
    c = 0; m = 0; list = ""
    for (r = 1; r <= rounds; r++) {
      ko = base SUBSEP d[i] SUBSEP r; kr = ref SUBSEP d[i] SUBSEP r
      if (!(ko in rn) || !(kr in rn)) continue
      q = (rs[ko] / rn[ko]) / (rs[kr] / rn[kr]); c++; v[c] = q; m += q
      list = list sprintf(" r%d %.4f", r, q)
    }
    if (c == 0) continue
    m /= c; ss = 0
    for (x = 1; x <= c; x++) ss += (v[x] - m) ^ 2
    if (c < 2) ci = "(one round: no interval)"
    else if (c - 1 > nt) ci = sprintf("(no t quantile for df %d)", c - 1)
    else ci = sprintf("± %.4f", t[c - 1] * sqrt(ss / (c - 1)) / sqrt(c))
    ao = base SUBSEP d[i]; ar = ref SUBSEP d[i]
    busy = tagged ? sprintf("  busy: %s [cpu-busy %d/%d] [other-busy %d/%d], %s [cpu-busy %d/%d] [other-busy %d/%d]", base, bc[ao], an[ao], bo[ao], an[ao], ref, bc[ar], an[ar], bo[ar], an[ar]) : ""
    printf "%s%-5s %s/%-6s  mean %.4f %s (n=%d)  of means %.4f  per round:%s%s\n", prefix, d[i], base, ref, m, ci, c, (as[ao] / an[ao]) / (as[ar] / an[ar]), list, busy
  }
}'
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  ref_witness | sed 's/^   /[dry]/'
  echo "[dry] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0} now: $(cpu_busy_reading)"
  for i in "${!ARMS[@]}"; do
    a=${ARMS[$i]}
    dep=${A_DEP[$i]}
    case ${A_KIND[$i]} in
      ref)
        ref_cmd "${A_ENG[$i]}" "$dep"
        echo "[dry] $a: timeout --kill-after=10 $BOUND env ${REF_ENV[*]} $REF_BIN ${REF_ARGS[*]}   # row label '${REF_LABEL% |}'${REF_BATCH:+, $REF_BATCH}"
        ;;
      *)
        note='' feedline="--depth $dep"
        [ "${A_LABEL[$i]}" = ours ] || note="   # row label '${A_LABEL[$i]}'"
        if [ "${A_KIND[$i]}" = prose ]; then
          feedline="--tokens \"\$(head -n $dep $PROSE | paste -sd, -)\""
          note="$note, $dep of the file's $PROSE_N ids, first ${A_TOK[$i]%%,*}, last ${A_TOK[$i]##*,}"
        fi
        arm_envs "$i"
        echo "[dry] $a: timeout --kill-after=10 $BOUND ${ARM_ENVS[*]:+env ${ARM_ENVS[*]} }${A_BIN[$i]} $feedline -n $N --time${WARM:+ --warm $WARM}$note"
        ;;
    esac
  done
  if [ "$AB_WARMUP" = 1 ]; then
    echo "[dry] warmup: ${ARMS[0]} once before round 1 (its command line above), discarded — its row prints as WARMUP r0 and is in no mean, ratio or row count (BLOOMERY_AB_WARMUP=0 skips it)"
  else
    echo "[dry] warmup: off (BLOOMERY_AB_WARMUP=0): round 1's first row is the lease's first process"
  fi
  for r in $(seq "$ROUNDS"); do
    order=()
    for i in $(seq 0 $((${#ARMS[@]} - 1))); do order+=("${ARMS[$(((i + r - 1) % ${#ARMS[@]}))]}"); done
    echo "[dry] round $r order: ${order[*]}"
  done
  exit 0
fi

lease_take
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s warmup=$AB_WARMUP"
echo "[config] ours: $BIN (placement (a), default ctx)"
[ -z "$PROSE_N" ] || echo "[config] prose: the first P ids of $PROSE ($PROSE_N ids)"
echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS env=$IK_GPU_ENV"
echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS (lcpp<K>: --n-cpu-moe K)"
echo "[config] prefill: ikpp/lcpppp run llama-bench -p P -n 0 -r 2 -o json at the flags above, the row is repetition 2 (<U>: -ub U -b max(U, 2048)); ours from its time prompt row"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0}"
witness pre
ref_witness
guard_other
guard_cpu pre

sums=() pp_sums=()
n_rows=0 busy_rows=0 other_rows=0
if [ "$AB_WARMUP" = 1 ]; then
  CPU_BUSY_TAG=
  guard_other
  guard_cpu "pre warmup ${ARMS[0]}"
  ROW_TAG=WARMUP
  case ${A_KIND[0]} in
    ref) ref_arm "${A_ENG[0]}" "${A_DEP[0]}" 0 ;;
    *) ours_arm 0 0 ;;
  esac
  ROW_TAG=ROW
  echo "[warmup] ${ARMS[0]} ran once before round 1 and is discarded (the WARMUP row above): the lease's first process reads the model's pages cold"
fi
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#ARMS[@]} - 1))); do
    j=$(((i + r - 1) % ${#ARMS[@]}))
    CPU_BUSY_TAG=
    guard_other
    guard_cpu "pre r$r ${ARMS[$j]}"
    case ${A_KIND[$j]} in
      ref) ref_arm "${A_ENG[$j]}" "${A_DEP[$j]}" "$r" ;;
      *) ours_arm "$j" "$r" ;;
    esac
  done
done
echo
echo "cpu-busy rows: $busy_rows of $n_rows (BLOOMERY_CPU_BUSY_PCT=${CPU_BUSY_PCT}% over [$CPU_BUSY_COMMS])"
echo "other-busy rows: $other_rows of $n_rows (a compute process on the other card as the arm started)"
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, the references"
echo "    llama-bench's own mean over the N steps — the cross-engine ratio reads these. The p50"
echo "    column is ours only. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  k = $1 " d=" $2; s[k] += $4; n[k]++; if ($5 != "") { sp[k] += $5; np[k]++ }
  if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
} END { for (k in s) {
  spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
  printf "mean %-14s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)\n", k, s[k] / n[k], mn[k], mx[k], spread, (np[k] ? sprintf("%.2f tok/s(p50)", sp[k] / np[k]) : ""), n[k] } }' | sort
echo
echo "=== ours / reference per depth: each round's ratio of the pair measured in that round, their"
echo "    mean with its 95 % interval (Student t, rounds - 1 degrees of freedom; 2.0 past 21 rounds),"
echo "    and the ratio of the arm means ==="
deps=$(printf '%s\n' "${A_DEP[@]}" | sort -un | tr '\n' ' ')
# The prose labels have their own tables: their prompt is not the one ours and the references ran.
refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -vx ours | grep -v '^prose' | sort -u | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$deps" "$refs" 0 ours
prose_refs=$(printf '%s\n' "${A_LABEL[@]}" | grep '^prose@' | sort -u | tr '\n' ' ')
if [ -n "$prose_refs" ]; then
  echo
  echo "=== the prose prompt: prose / each prose@ arm per P, the same statistics ==="
  printf '%s\n' "${sums[@]}" | ratio_table "ratio prose d=" "$deps" "$prose_refs" 0 prose
fi
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== prefill per prompt length (tok/s(pp) @ n=0, prompt P, $CARD_NAME). Ours: its time prompt"
  echo "    row (the P fed steps through token 0's readback); the references: llama-bench's pp value"
  echo "    over one repetition. The tags count the rows that met contention. ==="
  printf '%s\n' "${pp_sums[@]}" | awk -F'|' '{
    k = $1 " p=" $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
    if ($5 ~ /cpu-busy/) c[k]++; if ($5 ~ /other-busy/) o[k]++
  } END { for (k in s) {
    spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
    printf "mean pp %-14s %8.2f tok/s(pp)  [%s..%s, spread %.2f%%]  (n=%d)  [cpu-busy %d/%d] [other-busy %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, n[k], c[k], n[k], o[k], n[k] } }' | sort
  echo
  echo "=== ours / reference prefill per prompt length: the decode table's statistics over the pp"
  echo "    values, then how many of each side's rows carried [cpu-busy] and [other-busy] ==="
  pp_keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  pp_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -vx ours | grep -v '^prose' | sort -u | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$pp_keys" "$pp_refs" 1 ours
  pp_prose_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep '^prose@' | sort -u | tr '\n' ' ')
  if [ -n "$pp_prose_refs" ]; then
    echo
    echo "=== the prose prompt's prefill: prose / each prose@ arm per P, the same statistics ==="
    printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp prose p=" "$pp_keys" "$pp_prose_refs" 1 prose
  fi
fi
witness post
ref_witness
