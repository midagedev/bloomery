#!/usr/bin/env bash
# GLM-5.3-Flash decode by depth and prefill by prompt length on the timing card, three engines in one
# lease (run on the box, lead-only): our engine (generate_glm5next --time), llama.cpp on the two open
# PR branches that build glm5next (#27752, #27754; mainline does not) and exllamav3's own bench on its
# EXL3 quantization of the model.
#
#   just depth-gpu-glm5next 512 lcpp27752:512 lcpp27754:512
#   just depth-gpu-glm5next 512 lcpp27752pp:512 lcpp27754pp:512 exl3pp:512
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-glm5next 512 lcpp27754:512   # the command lines, no lease, no load
#
# depth-qwen3moe.sh's shape and tables; what differs is below. Every number carries its conditions:
# `tok/s @ n=N, depth D, <card>`, and the prefill `tok/s(pp) @ n=0, prompt P, <card>`.
#
# Arms:
#   <D>       ours: generate_glm5next --tokens <D ids of GLM_PROSE from GLM_PROSE_FROM> -n N --ctx C --place
#             PLACE --time. The prompt is fed one decode step a position (the program has no batched
#             prefill), so the row's pp_tok/s is the step feed's rate, `kind=steps`: D over the wall
#             from the first fed step to the readback of generated token 0. The N - 1 steps after
#             token 0 are timed. C is one context for every ours arm (BLOOMERY_GEN_CTX, default
#             2048): the plan refuses more positions than the latent layers attend whole (2,051), so
#             D + N <= C is checked before the lease. The prompt is prose under this model's own
#             vocabulary (models/glm5next.sh GLM_PROSE, its sha256 checked before the lease): the D
#             ids from 0-based index GLM_PROSE_FROM, past the GLM_HOT_TRACE ids the hot list was traced
#             from (docs/fair-measure.md 4.2: the list is held out from the measured prompt). Each ours,
#             hot and server row prints `prompt ids <a>..<b>`; a hot row whose ids overlap the trace
#             ends in ` [in-trace]`, the favourable case, which the contract publishes only as such.
#   hot:<D>   ours with BLOOMERY_HOT_LIST=$BLOOMERY_DATA/$GLM_HOT (the profile's list; a plain <D>
#             arm runs with BLOOMERY_HOT_LIST unset): the card experts ranked by the hot list. The row carries the plan's card_experts; a
#             hot row whose plan put no expert on the card is a FAIL row (a binary whose placement
#             keeps every routed expert on the host ignores the list), never an ours twin.
#   lcpp27752:<D>, lcpp27754:<D>   the PR branch's llama-bench -p 0 -n N -d D -r 1 at the profile's
#             LCPP27752_GPU_FLAGS / LCPP27754_GPU_FLAGS (the second under LCPP27754_ENV). -d prefills
#             D of llama-bench's own std::rand() ids before its clock starts; its row label is `tgN @
#             dD`.
#   lcpp27752pp:<P>, lcpp27754pp:<P>   the branch's prefill: llama-bench -p P -n 0 -r 1 at the same
#             flags, `ppP`. lcpp27752pp<U>:<P> (and 27754's) adds -ub U -b max(U, 2048): the ubatch
#             lever, not a default. P is not bounded by our context: the branches run the indexer
#             past 2,051 positions, ours does not, so a P above C has no ours row beside it.
#   lcpp27752fit:<D>, lcpp27754fit:<D>, lcpp27752ppfit[<U>]:<P>, lcpp27754ppfit[<U>]:<P>   the
#             branch at llama.cpp's own placement: the lcpp2775x:<D> and lcpp2775xpp[<U>]:<P> command
#             lines with the profile's placement options (-ngl, --n-cpu-moe, -ts, -ot) removed and
#             `-fitt 1024 -v` added, so llama-bench's fit chooses the layers and the overrides
#             (tools/ref/lcpp-fit.sh: what the fit does and where, the flags, the probe, the column);
#             #27754's twins keep LCPP27754_ENV and -fa off. The fit keeps the trailing blocks'
#             experts on the host where --n-cpu-moe keeps the leading ones' (models/glm5next.sh has
#             the predicted placement). The row carries `fit <what it chose>` from -v's loader
#             lines, echoed as `<label> fit …`. A fit that failed or never ran is a FAIL row naming
#             it, whatever the exit code: llama-bench ignores the fit's status and loads at -ngl -1
#             with no overrides, which on this file runs out of card memory or times a placement
#             nobody chose. Refused before the lease when that branch's llama-bench --help lists no
#             --fit-target, or when its flags already carry a fit option.
#             The server arms have no fit twin: an MTP context's fit counts the NextN block
#             (common/fit.cpp:141 in #27754, 140 in #27752, under load_mtp) and a server without the
#             draft's does not, so a fitted srv/mtp pair would sit at two placements and their ratio
#             would no longer be the draft's alone; the placement question is the llama-bench twins'.
#   lcpp27754mtp:<D>, lcpp27754srv:<D> (and lcpp27752's)   the branch's llama-server, because
#             llama-bench drives no speculation: one server process a row (LCPP2775x_SRV_FLAGS, or
#             LCPP2775x_MTP_FLAGS with the MTP draft, --spec-type draft-mtp --spec-draft-n-max 2, at
#             GLM_NCMOE_MTP), -c D + N + 256 rounded up to 256, on 127.0.0.1 at a free port; after
#             its /health answers (its own warm-up runs before that), one POST /completion of
#             ours' D prompt ids (from GLM_PROSE_FROM), so the draft sees prose — with n_predict N,
#             temperature 0, ignore_eos and cache_prompt off. The row is the response's `timings`:
#             predicted_per_second over predicted_n (the decode, drafts included), prompt_n and
#             prompt_per_second, and for the MTP arm draft_n / draft_n_accepted beside the server's
#             own `draft acceptance = … (A accepted / G generated), mean len = …` log line, which the
#             row carries verbatim. predicted_n other than N, or an MTP arm that drafted nothing, is a
#             FAIL row. The srv arm is the MTP arm's twin without the draft: the same binary, request
#             and path, so their ratio is the draft's alone.
#   exl3:<D>  exllamav3's eval/perf.py -spf --max_length D + 256 at EXL3_FLAGS on EXL3_MODEL, the
#             `Context D` row of its Generation table: 100 steps at depth D (its fixed count, the
#             row says n=100) over wikitext-2 ids, the recurrent state a test state of that depth.
#             D is 0 or a multiple of 256 (perf.py's lengths). exl3pp:<P> is perf.py -sg --max_length
#             P, the `Length P` row of its Prefill table: P ids in chunks of its default 4096, P a
#             multiple of 256 up to 4096. Another quantization: its rows are in the means and in a
#             table of their own, never in a ratio against ours.
# Every arm is one process: a load, the prefill, the steps. The references open the model through a
# file mapping; ours reads its host set at load (MADV_POPULATE_READ), outside its timer.
#
# Paging. The GGUF split set (199.7 GB) fits in the page cache (about 245 GB on this box), ours and
# the branches read the same file pages, so their arms rotate freely. The EXL3 directory (154 GB) and
# its CPU-tier copies do not fit beside it: the exllamav3 arms run as a block after every GGUF round,
# opened by a discarded process (`DISCARD r0`), so no GGUF row is timed after the EXL3 set evicted the
# file. Every GGUF arm runs once, discarded, right before its round-1 row, on the same ids (`WARMUP
# r0`, docs/fair-measure.md 2.1; the exllamav3 block keeps its one discard; BLOOMERY_AB_WARMUP=0 skips
# both): the sitting's earlier segments leave another model's pages cached.
# The file does not stay whole in the cache on its own: our load drops the file pages of what it
# uploads (BLOOMERY_CARD_DONTNEED, 1 by default: the card trunk, 8.97 GB, and the card experts,
# 39.94 GB), so the next arm whose host set holds those pages reads them from the drive — ours in its
# populate, before its timer (a hot arm after an id-prefix arm: the id-prefix card experts the hot
# list leaves on the host), llama-bench through its mapping, at its load and inside its timer.
# Preheat. Before every GGUF arm (the warm-up included; no exllamav3 arm) the runner reads a host set
# into the page cache with tools/ref/gguf-ranges.py (host, once per K before the lease; preheat, pread
# in chunks, the data discarded): token_embd and the routed experts of blocks 0..K-1. A llama.cpp
# arm's K is its own --n-cpu-moe (the MTP arm's GLM_NCMOE_MTP); ours' and a fit twin's is the
# profile's GLM_PREHEAT_K, every routed expert of the engine's blocks, a superset of ours' host set
# under any card rule and of the fit's trailing blocks' routed experts — gguf-ranges.py cannot read a
# plan, and the whole set fits beside the rest of the file. The read sits outside the arm's witness
# blocks, its wall and its fault counts, and prints `preheat <label> K=<k> bytes=<b> s=<t>
# gbps=<rate>`: the rate tells how much the arm before left uncached. For ours it moves the drive
# reads out of the load and makes the load's faults independent of the arm before; it cannot move the
# timed window, which starts after the populate. Flags whose host set the rule does not model (-ot,
# --override-tensor, -cmoe, --cpu-moe, a list for --n-cpu-moe) are refused before the lease.
# BLOOMERY_PREHEAT=0 turns the preheat off (1, the default; anything else is refused). A dry run
# prints each arm's K and the `host` line of every K, and reads nothing but the headers.
# Every row carries `majflt <n>` and ` [cold]` when those faults at COLD_US µs each (75.4: the serial
# 4 KB fault measured on this box, rig-log 2026-09-23, depth-ds41.sh's price) could be 1 % of the
# row's timed window or more. Ours counts from its `fed` record, which generate_glm5next prints just
# before its feed's timer starts, to its exit (majflt_mark, tools/ref/cold-blocks.sh), and prints the
# whole process's count beside it; an ours output with no `fed` line counts the whole process and says
# so. Every other row counts the change in /proc/vmstat pgmajfault across the arm's whole process
# (load and warm-up included, so an upper bound on its timed window's). exllamav3 rows print their
# count untagged: its load reads the 154 GB directory and copies the CPU experts into its own memory,
# faulting in every process, and perf.py's clock starts after that load, reading no file.
# A round's row that reads [cold] prints as `COLD r<r> …`, is in no mean, and its arm runs once more
# right away (its preheat included); a second [cold] prints `FAIL-cold r<r> …`, stays out of the means
# and the ratios, and joins the failed list (docs/fair-measure.md 2.3). Warm-up rows are not re-run.
#
# Contention. Before every arm: the other card (guard_other, ` [other-busy]`), the timing card
# (guard_timing: waits for another round's process on it, rc 75 after 10 minutes) and the CPU
# (guard_cpu, before and after, ` [cpu-busy]`: every engine here runs routed experts on the host
# cores). BLOOMERY_OTHER_STRICT=1 aborts on either tag instead.
#
# Failures. An arm that exits non-zero, prints no value, or (hot) loaded no card expert prints `FAIL
# r<r> <label> <d|p>=<X> rc=<rc> | <why> | full output: <file>` where its row would be; the runner
# goes on, that label at that key drops out of the means and ratios, and the runner exits 1 at the end
# with the list.
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (default 2), BLOOMERY_GEN_WARM
# (--warm), BLOOMERY_GEN_CTX (C), BLOOMERY_GEN_BIN (default target/release/generate_glm5next),
# BLOOMERY_GEN_PLACE (a, the default, needs the A6000 as the timing card; gate the 3090),
# BLOOMERY_AB_WARMUP (1 or 0), BLOOMERY_PREHEAT (1 or 0, above), BLOOMERY_ARM_BOUND (seconds one
# arm, or one preheat, may run, default 900), BLOOMERY_DRY=1 (every arm's command line and preheat,
# the trees, the checks and each round's order, then exit 0 before the lease: nothing is loaded and
# nothing is timed).
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = glm5next ] || {
  echo "depth-glm5next.sh: the profile is $MODEL_NAME — BLOOMERY_MODEL=glm5next on the Mac side" >&2
  exit 64
}
RECORDS="${BASH_SOURCE[0]%/*}/../bloomery/records.py"
N=${BLOOMERY_DECODE_N:-96}
ROUNDS=${BLOOMERY_AB_ROUNDS:-2}
WARM=${BLOOMERY_GEN_WARM:-}
CTX=${BLOOMERY_GEN_CTX:-2048}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_glm5next}
PLACE=${BLOOMERY_GEN_PLACE:-a}
BOUND=${BLOOMERY_ARM_BOUND:-900}
WARMUP=${BLOOMERY_AB_WARMUP:-1}
PREHEAT=${BLOOMERY_PREHEAT:-1}
# majflt_mark (the timed window's fault mark) is cold-blocks.sh's; sourced before COLD_US, which this
# runner keeps at its own value.
# shellcheck source=tools/ref/cold-blocks.sh
source "${BASH_SOURCE[0]%/*}/cold-blocks.sh" || exit 2
COLD_US=75.4
DRY=${BLOOMERY_DRY:-}
PROSE=$BLOOMERY_DATA/$GLM_PROSE
PROSE_FROM=$GLM_PROSE_FROM
HOT=$BLOOMERY_DATA/$GLM_HOT
case $PROSE_FROM:$GLM_HOT_TRACE in *[!0-9:]* | :* | *:) echo "depth-glm5next.sh: GLM_PROSE_FROM and GLM_HOT_TRACE are id counts, got '$PROSE_FROM' and '$GLM_HOT_TRACE'" >&2; exit 64 ;; esac
for v in N:$N ROUNDS:$ROUNDS CTX:$CTX BOUND:$BOUND; do
  case ${v#*:} in '' | *[!0-9]* | 0*) echo "depth-glm5next.sh: ${v%%:*} is a positive integer, got '${v#*:}'" >&2; exit 64 ;; esac
done
case $PLACE in a | gate) ;; *) echo "depth-glm5next.sh: BLOOMERY_GEN_PLACE is a or gate, got '$PLACE'" >&2; exit 64 ;; esac
case $WARMUP in 0 | 1) ;; *) echo "depth-glm5next.sh: BLOOMERY_AB_WARMUP is 0 or 1, got '$WARMUP'" >&2; exit 64 ;; esac
case $PREHEAT in 0 | 1) ;; *) echo "depth-glm5next.sh: BLOOMERY_PREHEAT is 1 (read each GGUF arm's host set before it, the default) or 0, got '$PREHEAT'" >&2; exit 64 ;; esac
case $WARM in '' | [0-9] | [1-9][0-9]*) ;; *) echo "depth-glm5next.sh: BLOOMERY_GEN_WARM is a count, got '$WARM'" >&2; exit 64 ;; esac

ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(512 lcpp27754:512)
# Per arm: the engine (ours, hot, lcpp27752, lcpp27754, exl3), whether it is a prefill arm, its
# ubatch lever, its depth or prompt length, and its row label (a prefill arm's names its ubatch).
A_ENG=() A_PP=() A_UB=() A_DEP=() A_LABEL=()
ours=0 lcpp=0 exl3=0 gguf=0 srv=0 fit27752=0 fit27754=0
usage() {
  echo "depth-glm5next.sh: arm '$1' is <D>, hot:<D>, lcpp27752[fit]:<D>, lcpp27754[fit]:<D>, lcpp27752pp[fit][<U>]:<P>, lcpp27754pp[fit][<U>]:<P>, lcpp2775{2,4}{srv,mtp}:<D>, exl3:<D> or exl3pp:<P>${2:+ — $2}" >&2
  exit 64
}
# The fit arms' flags, probe and column (lcpp-fit.sh); fit_eng names this runner's fit engines.
# shellcheck source=tools/ref/lcpp-fit.sh
source "${BASH_SOURCE[0]%/*}/lcpp-fit.sh" || exit 2
fit_eng() { case $1 in lcpp2775[24]fit | lcpp2775[24]ppfit | lcpp2775[24]ppfit[1-9]*) return 0 ;; *) return 1 ;; esac; }
# gpu_flags <engine>: the profile's llama-bench flags of that engine's branch.
gpu_flags() { case $1 in lcpp27752*) echo "$LCPP27752_GPU_FLAGS" ;; *) echo "$LCPP27754_GPU_FLAGS" ;; esac; }
for a in "${ARMS[@]}"; do
  eng=${a%%:*} dep=${a#*:} pp=0 ub=''
  [ "$a" != "$eng" ] || { eng=ours dep=$a; }
  case $dep in '' | *[!0-9]*) usage "$a" ;; esac
  case $eng in
    ours | hot)
      ours=1
      [ "$dep" -ge 1 ] && [ $((dep + N)) -le "$CTX" ] || usage "$a" "ours needs 1 <= D and D + N <= C ($N + D against --ctx $CTX)"
      ;;
    lcpp27752 | lcpp27754) lcpp=1 ;;
    lcpp27752fit | lcpp27754fit) lcpp=1 ;;
    lcpp27752srv | lcpp27754srv | lcpp27752mtp | lcpp27754mtp)
      lcpp=1 srv=1
      [ "$dep" -ge 1 ] || usage "$a" "a server arm feeds D >= 1 prose ids"
      ;;
    lcpp27752pp* | lcpp27754pp*)
      lcpp=1 pp=1 ub=${eng#lcpp2775?pp}
      ub=${ub#fit}
      case $ub in *[!0-9]* | 0*) usage "$a" ;; esac
      [ "$dep" -ge 1 ] || usage "$a" "a prompt of 0 ids has no prefill to time"
      ;;
    exl3)
      exl3=1
      [ $((dep % 256)) = 0 ] || usage "$a" "perf.py measures depth 0 and multiples of 256"
      ;;
    exl3pp)
      exl3=1 pp=1
      [ "$dep" -ge 256 ] && [ "$dep" -le 4096 ] && [ $((dep % 256)) = 0 ] || usage "$a" "perf.py measures prompts of multiples of 256, up to one 4096 chunk here"
      ;;
    *) usage "$a" ;;
  esac
  if fit_eng "$eng"; then
    lcpp_fit_flags "$(gpu_flags "$eng")" || usage "$a" "$FIT_WHY"
    case $eng in lcpp27752*) fit27752=1 ;; *) fit27754=1 ;; esac
  fi
  case $eng in exl3*) ;; *) gguf=1 ;; esac
  A_ENG+=("$eng") A_PP+=("$pp") A_UB+=("$ub") A_DEP+=("$dep") A_LABEL+=("$eng")
done

# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
CPU_BUSY_COMMS=${BLOOMERY_CPU_BUSY_COMMS:-$CPU_BUSY_COMMS generate_glm5next generate_qwen3moe python3}
WITNESS=(head-open indent card busiest model mem pgmajfault)
T975=$(python3 "${BASH_SOURCE[0]%/*}/tdist.py" "$ROUNDS") || {
  echo "depth-glm5next.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS" >&2
  exit 2
}
# Plan (a) is made for the card named A6000, the gate plan for the 3090; the arms see the timing card only.
if [ "$ours" = 1 ]; then
  want=$GPU_A6000
  [ "$PLACE" = a ] || want=$GPU_3090
  [ "$TIMING_GPU" = "$want" ] || {
    echo "depth-glm5next.sh: --place $PLACE loads on the $([ "$PLACE" = a ] && echo A6000 || echo 3090), but the timing card is $TIMING_GPU (BLOOMERY_TIMING_GPU)" >&2
    exit 64
  }
fi
CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')

# What every arm needs, checked before the lease (a dry run prints the findings and goes on). Ours:
# a binary no older than its sources that declares the timing records, the prose ids at their sha256,
# the hot list for a hot arm. The references: their binaries, and exllamav3's venv, bench and model.
CHECKS=()
check() { # check <rc> <message>: fatal before a real run, printed in a dry one
  if [ -n "$DRY" ]; then CHECKS+=("$2"); else echo "depth-glm5next.sh: $2" >&2; exit "$1"; fi
}
if [ "$ours" = 1 ] || [ "$srv" = 1 ]; then
  if [ ! -f "$PROSE" ]; then
    check 66 "no prose ids at $PROSE"
  elif [ "$(sha256sum "$PROSE" | cut -d' ' -f1)" != "$GLM_PROSE_SHA256" ]; then
    check 65 "$PROSE is not the prose the profile pins (sha256 $GLM_PROSE_SHA256)"
  else
    for i in "${!ARMS[@]}"; do
      case ${A_ENG[$i]} in exl3* | lcpp2775[24] | lcpp2775[24]fit | lcpp2775[24]pp*) continue ;; esac
      [ $((PROSE_FROM + A_DEP[i])) -le "$(wc -l < "$PROSE")" ] || check 64 "arm ${ARMS[$i]}: $PROSE holds fewer than GLM_PROSE_FROM + ${A_DEP[$i]} = $((PROSE_FROM + A_DEP[i])) ids"
    done
  fi
fi
if [ "$ours" = 1 ]; then
  if [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
  if [ -x "$BIN" ]; then
    kinds=$("$BIN" --records-schema 2> /dev/null | python3 -c 'import sys, json; print(" ".join(json.loads(l).get("kind", "") for l in sys.stdin if l.strip()))')
    for k in time_prompt time_step smoke plan; do
      case " $kinds " in *" $k "*) ;; *) check 2 "$BIN declares no '$k' record (--records-schema): it has no --time, so no ours arm can be timed" ;; esac
    done
  else
    check 2 "no binary at $BIN"
  fi
  for e in "${A_ENG[@]}"; do [ "$e" != hot ] || [ -f "$HOT" ] || { check 66 "hot arm: no hot list at $HOT"; break; }; done
fi
case " ${A_ENG[*]}" in *" lcpp27752"*) [ -x "$LCPP27752BIN" ] || check 2 "no llama-bench at $LCPP27752BIN (PR #27752's tree)" ;; esac
case " ${A_ENG[*]}" in *" lcpp27754"*) [ -x "$LCPP27754BIN" ] || check 2 "no llama-bench at $LCPP27754BIN (PR #27754's tree)" ;; esac
# A fit arm needs its branch's llama-bench to have the fit (lcpp_fit_probe runs its --help with no card).
# shellcheck disable=SC2153 # LCPP27752 and LCPP27754 are the profile's
{ [ "$fit27752" = 0 ] || [ ! -x "$LCPP27752BIN" ] || lcpp_fit_probe "$LCPP27752BIN"; } || check 64 "the lcpp27752fit/lcpp27752ppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP27752)"
{ [ "$fit27754" = 0 ] || [ ! -x "$LCPP27754BIN" ] || lcpp_fit_probe "$LCPP27754BIN"; } || check 64 "the lcpp27754fit/lcpp27754ppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP27754)"
case " ${A_ENG[*]} " in *" lcpp27752srv "* | *" lcpp27752mtp "*) [ -x "$LCPP27752SRV" ] || check 2 "no llama-server at $LCPP27752SRV" ;; esac
case " ${A_ENG[*]} " in *" lcpp27754srv "* | *" lcpp27754mtp "*) [ -x "$LCPP27754SRV" ] || check 2 "no llama-server at $LCPP27754SRV" ;; esac
[ "$srv" = 0 ] || command -v curl > /dev/null || check 2 "no curl for the server arms"
if [ "$exl3" = 1 ]; then
  [ -x "$EXL3_PY" ] || check 2 "no exllamav3 python at $EXL3_PY"
  # shellcheck disable=SC2153 # EXL3 is the profile's
  [ -f "$EXL3/eval/perf.py" ] || check 2 "no bench at $EXL3/eval/perf.py"
  [ -f "$EXL3_MODEL/config.json" ] || check 2 "no EXL3 model at $EXL3_MODEL"
  [ -f "$EXL3_WIKITEXT" ] || check 2 "no wikitext-2 test text at $EXL3_WIKITEXT"
fi

# A binary's sha256 and its tree's HEAD and dirty count. GIT_OPTIONAL_LOCKS=0 keeps `git status` from
# rewriting the index of a tree this root process does not own.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" 2> /dev/null | cut -c1-12)
  head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=10 HEAD 2> /dev/null || echo '?')
  dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  echo "$bin sha256=${sha:-?} head=$head dirty_files=$dirty"
}
REF_LINES=()
for e in lcpp27752 lcpp27754; do
  case " ${A_ENG[*]} " in *" $e"*) ;; *) continue ;; esac
  if [ "$e" = lcpp27752 ]; then REF_LINES+=("$e: $(tree_line "$LCPP27752BIN" "$LCPP27752")"); else REF_LINES+=("$e: $(tree_line "$LCPP27754BIN" "$LCPP27754")"); fi
done
if [ "$exl3" = 1 ]; then
  v=$(find "${EXL3_PY%/bin/*}"/lib/python3*/site-packages -maxdepth 1 -name 'exllamav3-*.dist-info' 2> /dev/null | sed 's/.*exllamav3-//; s/\.dist-info$//' | head -n 1)
  REF_LINES+=("exl3: $EXL3_PY exllamav3 ${v:-?} bench $(tree_line "$EXL3/eval/perf.py" "$EXL3") model $EXL3_MODEL ($EXL3_BPW bpw)")
fi
ref_witness() { [ ${#REF_LINES[@]} -eq 0 ] || printf '    %s\n' "${REF_LINES[@]}"; }

# prompt_ids <D>: the D prompt ids, one a line: GLM_PROSE from 0-based index PROSE_FROM (the header's
# Arms). prompt_col <engine> <D>: the row's `prompt ids` column, and ` [in-trace]` for a hot arm whose
# ids overlap the hot list's trace, ids 0..GLM_HOT_TRACE-1.
prompt_ids() { tail -n +"$((PROSE_FROM + 1))" "$PROSE" | head -n "$1"; }
prompt_col() {
  PROMPT_COL=" | prompt ids $PROSE_FROM..$((PROSE_FROM + $2 - 1))" TRACE_TAG=''
  [ "$1" != hot ] || [ "$PROSE_FROM" -ge "$GLM_HOT_TRACE" ] || TRACE_TAG=' [in-trace]'
}
# The command of arm <i>, into CMD (an array, its environment first through env), LABEL_TEST
# (the row the reference's output is read by) and, for a fit arm, FIT_NOTE (what its flags dropped).
arm_cmd() {
  local i=$1 eng=${A_ENG[$1]} dep=${A_DEP[$1]} ub=${A_UB[$1]} flags envs=''
  CMD=() LABEL_TEST='' BATCH='' FIT_NOTE=''
  case $eng in
    ours | hot)
      envs="-u BLOOMERY_HOT_LIST"
      [ "$eng" = ours ] || envs="BLOOMERY_HOT_LIST=$HOT"
      # shellcheck disable=SC2206 # an empty envs adds nothing
      CMD=(env $envs "$BIN" --tokens "$(prompt_ids "$dep" | paste -sd, -)" -n "$N" --ctx "$CTX" --place "$PLACE" --time ${WARM:+--warm "$WARM"})
      ;;
    lcpp2775[24]srv | lcpp2775[24]mtp)
      # shellcheck disable=SC2206 # the profile's NAME=VALUE words
      case $eng in
        lcpp27752srv) CMD=(env "$LCPP27752SRV") flags=$LCPP27752_SRV_FLAGS ;;
        lcpp27752mtp) CMD=(env "$LCPP27752SRV") flags=$LCPP27752_MTP_FLAGS ;;
        lcpp27754srv) CMD=(env $LCPP27754_ENV "$LCPP27754SRV") flags=$LCPP27754_SRV_FLAGS ;;
        lcpp27754mtp) CMD=(env $LCPP27754_ENV "$LCPP27754SRV") flags=$LCPP27754_MTP_FLAGS ;;
      esac
      # shellcheck disable=SC2206
      CMD=(timeout --kill-after=10 "$BOUND" "${CMD[@]}" -m "$MODEL" --host 127.0.0.1 -c "$(((dep + N + 256 + 255) / 256 * 256))" $flags)
      LABEL_TEST="POST /completion: GLM_PROSE ids $PROSE_FROM..$((PROSE_FROM + dep - 1)), n_predict $N, temperature 0, ignore_eos"
      ;;
    lcpp*)
      # shellcheck disable=SC2206 # the profile's NAME=VALUE words
      case $eng in lcpp27752*) CMD=(env "$LCPP27752BIN") flags=$LCPP27752_GPU_FLAGS ;; *) CMD=(env $LCPP27754_ENV "$LCPP27754BIN") flags=$LCPP27754_GPU_FLAGS ;; esac
      if fit_eng "$eng"; then
        lcpp_fit_flags "$flags"
        flags=$FIT_FLAGS
        FIT_NOTE="placement: llama-bench's fit at -fitt $LCPP_FIT_TARGET MiB (dropped: $FIT_DROPPED), -v for the fit column"
      fi
      if [ "${A_PP[$i]}" = 1 ]; then
        CMD+=(-m "$MODEL" -p "$dep" -n 0 -r 1) LABEL_TEST="pp$dep |" BATCH="ub 512 b 2048 (llama-bench defaults)"
        if [ -n "$ub" ]; then
          CMD+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))") BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
        fi
      else
        CMD+=(-m "$MODEL" -p 0 -n "$N" -d "$dep" -r 1) LABEL_TEST="tg$N @ d$dep |"
      fi
      # shellcheck disable=SC2206 # the profile keeps the flags as one string
      CMD+=($flags)
      ;;
    exl3)
      # shellcheck disable=SC2206
      CMD=(sudo -u user env --chdir=/ "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$EXL3_PY" "$EXL3/eval/perf.py" -m "$EXL3_MODEL" -spf --max_length "$((dep + 256))" $EXL3_FLAGS)
      LABEL_TEST="Context $dep:"
      ;;
    exl3pp)
      # shellcheck disable=SC2206
      CMD=(sudo -u user env --chdir=/ "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$EXL3_PY" "$EXL3/eval/perf.py" -m "$EXL3_MODEL" -sg --max_length "$dep" $EXL3_FLAGS)
      LABEL_TEST="Length $dep:"
      ;;
  esac
}
# exllamav3 runs as the tree's owner, from / (its CPU-tier worker process opens the working directory,
# which that user cannot enter under /root): perf.py keeps a disk cache beside itself, and its token stream
# reads the wikitext-2 test text from the temp dir, downloading it when absent; the staged copy
# keeps the network out of the lease.
stage_exl3() {
  local d=/tmp/llama_cpp_ppl_wikitext2/wikitext-2-raw
  [ -f "$d/wiki.test.raw" ] && return 0
  sudo -u user mkdir -p "$d" && sudo -u user cp "$EXL3_WIKITEXT" "$d/wiki.test.raw"
}

# The order: GGUF arms rotated by one slot each round, all their rounds, then the EXL3 arms likewise.
order_of() { # order_of <round> <exl3 0|1>: the indices of that group's arms in that round's order
  local r=$1 want=$2 i
  local -a idx=()
  for i in "${!ARMS[@]}"; do
    case ${A_ENG[$i]} in exl3*) [ "$want" = 1 ] && idx+=("$i") ;; *) [ "$want" = 0 ] && idx+=("$i") ;; esac
  done
  local n=${#idx[@]} k
  for ((k = 0; k < n; k++)); do printf '%s ' "${idx[$(((k + r - 1) % n))]}"; done
}
first_of() { local o; o=$(order_of 1 "$1"); echo "${o%% *}"; }

# The preheat's plan, before the lease (the header's Preheat): per arm A_PHK, the K of the host set it
# reads (empty for an exllamav3 arm, with BLOOMERY_PREHEAT=0, or in a dry run whose ranges failed), and
# per K and -ngl gguf-ranges.py's `host` line PH_LINE_<K>_<ngl> and the ranges file PH_DIR/k<K>.tsv.
# PH_RATE prices a cold preheat in a dry run: GB/s, depth-ds41.sh's NVMe populate rate, the slow end.
GGUF_RANGES="${BASH_SOURCE[0]%/*}/gguf-ranges.py"
PH_RATE=1.33
PH_DIR='' A_PHK=()
# ph_k <index>: arm <index>'s host set K and -ngl into PHK and PHNGL (PHK empty: no preheat). Flags whose
# host set the rule does not model exit 64.
ph_k() {
  local i=$1 w prev='' k=0 ngl='' fit=0
  PHK='' PHNGL=''
  case ${A_ENG[$i]} in
    exl3*) return 0 ;;
    ours | hot) PHK=$GLM_PREHEAT_K; return 0 ;;
  esac
  arm_cmd "$i"
  for w in "${CMD[@]}"; do
    case $w in
      -ot | --override-tensor | --override-tensor=* | -cmoe | --cpu-moe)
        echo "depth-glm5next.sh: arm '${ARMS[$i]}': its flags carry $w, a host set the preheat does not model; set BLOOMERY_PREHEAT=0 to run it unpreheated" >&2
        exit 64
        ;;
      -fitt | --fit-target) fit=1 ;;
    esac
    case $prev in
      --n-cpu-moe | -ncmoe) k=$w ;;
      -ngl | --n-gpu-layers | --gpu-layers) ngl=$w ;;
    esac
    prev=$w
  done
  if [ "$fit" = 1 ]; then
    PHK=$GLM_PREHEAT_K
    return 0
  fi
  case $k in '' | *[!0-9]*) echo "depth-glm5next.sh: arm '${ARMS[$i]}': --n-cpu-moe '$k' is not one layer count (a list runs several placements)" >&2; exit 64 ;; esac
  case $ngl in *[!0-9]*) echo "depth-glm5next.sh: arm '${ARMS[$i]}': -ngl '$ngl' is not one layer count" >&2; exit 64 ;; esac
  PHK=$k PHNGL=$ngl
}
if [ "$PREHEAT" = 1 ] && [ "$gguf" = 1 ]; then
  PH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/depth-glm5next-preheat.XXXXXX") || exit 2
  trap 'rm -rf "$PH_DIR"' EXIT
  for i in "${!ARMS[@]}"; do
    ph_k "$i"
    A_PHK[i]=$PHK
    [ -n "$PHK" ] || continue
    var=PH_LINE_${PHK}_${PHNGL:-none}
    [ -z "${!var:-}" ] || continue
    # shellcheck disable=SC2086 # an empty PHNGL adds nothing
    if line=$(python3 "$GGUF_RANGES" host "$MODEL" --n-cpu-moe "$PHK" ${PHNGL:+--ngl "$PHNGL"} --out "$PH_DIR/k$PHK.tsv"); then
      printf -v "$var" '%s' "$line"
    else
      rc=$?
      check "$rc" "arm '${ARMS[$i]}': no preheat ranges for K=$PHK (tools/ref/gguf-ranges.py rc $rc)"
      A_PHK[i]=''
    fi
  done
fi
# ph_bytes <K> <ngl>: the host set's bytes from its `host` line.
ph_bytes() {
  local var=PH_LINE_${1}_${2:-none} w
  for w in ${!var}; do case $w in bytes=*) echo "${w#bytes=}" ;; esac; done
}
# preheat_arm <index>: the host set of arm <index>'s K into the page cache (gguf-ranges.py preheat, under
# the arm bound) and its `preheat` line; non-zero with FAIL_WHY on a failure. Nothing without a plan.
preheat_arm() {
  local k=${A_PHK[$1]:-} out rc=0
  [ -n "$PH_DIR" ] && [ -n "$k" ] || return 0
  out=$(timeout --kill-after=10 "$BOUND" python3 "$GGUF_RANGES" preheat "$PH_DIR/k$k.tsv" 2>&1) || rc=$?
  if [ "$rc" -ne 0 ]; then
    FAIL_WHY="its preheat failed (rc $rc): ${out##*$'\n'}"
    return "$rc"
  fi
  echo "preheat ${A_LABEL[$1]} K=$k $out"
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS ctx=$CTX place=$PLACE warm=${WARM:-0} card=$CARD_NAME timing_gpu=$TIMING_GPU arm_bound=${BOUND}s warmup=$WARMUP cold_us=$COLD_US"
  echo "[dry] ours: $BIN prose=$PROSE hot=$HOT"
  echo "[dry] prompt: GLM_PROSE ids from index $PROSE_FROM; the hot list's trace: ids 0..$((GLM_HOT_TRACE - 1))"
  ref_witness | sed 's/^   /[dry]/'
  [ ${#CHECKS[@]} -eq 0 ] || printf '[dry] check: %s\n' "${CHECKS[@]}"
  for i in "${!ARMS[@]}"; do
    arm_cmd "$i"
    row=${LABEL_TEST% |}
    echo "[dry] ${ARMS[$i]}:${row:+ row \"$row\"}${BATCH:+, $BATCH}${FIT_NOTE:+, $FIT_NOTE}"
    case ${A_ENG[$i]} in lcpp2775[24]srv | lcpp2775[24]mtp) pre='' post=' --port <free>' ;; *) pre="timeout --kill-after=10 $BOUND " post='' ;; esac
    echo "[dry]     $pre$(printf '%q ' "${CMD[@]}" | sed -E 's/--tokens [^ ]+/--tokens <GLM_PROSE ids '"$PROSE_FROM..$((PROSE_FROM + A_DEP[i] - 1))"'>/')$post"
    [ -n "${A_PHK[$i]:-}" ] || continue
    ph_k "$i"
    b=$(ph_bytes "$PHK" "$PHNGL")
    ph_round=$((${ph_round:-0} + b))
    echo "[dry]     preheat K=$PHK: $b B ($(awk -v b="$b" 'BEGIN { printf "%.1f", b / 1e9 }') GB), $(awk -v b="$b" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s if all of it is cold at $PH_RATE GB/s"
  done
  if [ -n "$PH_DIR" ]; then
    for var in ${!PH_LINE_*}; do echo "[dry] preheat: ${!var}"; done
    echo "[dry] preheat: ${ph_round:-0} B a round over the GGUF arms, $(awk -v b="${ph_round:-0}" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s a round if every byte is cold at $PH_RATE GB/s (the upper end: a set the arm before left cached reads at page-cache speed)"
  else
    echo "[dry] preheat: off ($([ "$PREHEAT" = 0 ] && echo BLOOMERY_PREHEAT=0 || echo 'no GGUF arm'))"
  fi
  [ "$gguf" = 0 ] || [ "$WARMUP" = 0 ] || echo "[dry] WARMUP r0: every GGUF arm once on its own ids, discarded, right before its round-1 row"
  for r in $(seq "$ROUNDS"); do
    o='' ; for i in $(order_of "$r" 0); do o+="${ARMS[$i]} "; done
    [ "$gguf" = 0 ] || echo "[dry] round $r gguf order: $o"
  done
  [ "$exl3" = 0 ] || [ "$WARMUP" = 0 ] || echo "[dry] DISCARD r0: ${ARMS[$(first_of 1)]} (discarded, opens the exllamav3 block)"
  for r in $(seq "$ROUNDS"); do
    o='' ; for i in $(order_of "$r" 1); do o+="${ARMS[$i]} "; done
    [ "$exl3" = 0 ] || echo "[dry] round $r exl3 order: $o"
  done
  exit 0
fi

[ "$exl3" = 0 ] || stage_exl3 || { echo "depth-glm5next.sh: could not stage $EXL3_WIKITEXT for perf.py" >&2; exit 2; }

# A compute process on the timing card as an arm starts: another round's functional run. Every arm
# loads onto that card, so wait for it, polling every 10 s, and stop the runner after 10 minutes
# (rc 75: contention, not a result).
guard_timing() {
  local apps i
  for ((i = 0; i < 60; i++)); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * 10)) s" >&2
      return 0
    fi
    [ "$i" != 0 ] || echo "[timing-busy] $(now) compute apps on the timing card: [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
    sleep 10
  done
  witness abort-timing >&2
  exit 75
}
majflt() { awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat; }
strip() { sed 's/\x1b\[[0-9;]*m//g' | tr '\r' '\n'; }
# cold_col <faults> <timed window, s> [<what the count spans>]: the majflt column, and COLD_TAG when the
# faults could cost 1 % of the window. The span defaults to the whole process.
cold_col() {
  local f=$1 w=$2 pct
  pct=$(awk -v f="$f" -v us="$COLD_US" -v w="$w" 'BEGIN { printf "%.1f", (w > 0) ? 100 * f * us / 1e6 / w : 0 }')
  COLD_TAG=''
  awk -v p="$pct" 'BEGIN { exit !(p >= 1) }' && COLD_TAG=' [cold]'
  MAJ_COL=" | majflt $f (${3:-whole process}) <= $pct % of the timed window"
}

sums=() pp_sums=() failed=()
n_rows=0 busy_rows=0 other_rows=0 cold_rows=0 retry_rows=0 fail_cold=0
# cold_pass <tag> <round> <index>: PTAG, the word the row prints under the cold rule (the header):
# <tag> for a row that is not [cold] or not a round's; COLD on a round row's first pass, which sets
# COLD_AGAIN so run_row runs the arm once more; FAIL-cold on its second, which joins the failed list.
# Only a ROW row goes into the sums.
COLD_PASS=0 COLD_AGAIN=0
cold_pass() {
  local key=d
  PTAG=$1
  [ "$1" = ROW ] && [ -n "$COLD_TAG" ] || return 0
  if [ "$COLD_PASS" = 0 ]; then
    PTAG=COLD COLD_AGAIN=1
  else
    [ "${A_PP[$3]}" = 0 ] || key=p
    PTAG=FAIL-cold
    failed+=("r$2:${A_LABEL[$3]}@$key=${A_DEP[$3]}(cold)")
    fail_cold=$((fail_cold + 1))
  fi
}
# fail_row <tag> <round> <index> <rc> <why> <output>
fail_row() {
  local f=${TMPDIR:-/tmp}/depth-glm5next-${A_LABEL[$3]}-${A_DEP[$3]}-r$2.log key=d
  [ "${A_PP[$3]}" = 0 ] || key=p
  echo "$6" > "$f"
  echo "FAIL r$2 ${A_LABEL[$3]} $key=${A_DEP[$3]} rc=$4 | $5 | full output: $f$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  echo "$6" | tail -n 12 | sed 's/^/    /' >&2
  failed+=("r$2:${A_LABEL[$3]}@$key=${A_DEP[$3]}")
}

# srv_run <depth>: a server arm's process (CMD, which starts with its own timeout) on a free port, one
# request after /health answers, then the process stopped by the pid it was started with. Sets out (the
# server's log, then `RESPONSE <json>`) and rc (the server's exit when it died first, 124 when /health
# never answered inside BOUND, curl's code when the request failed).
srv_run() {
  local dep=$1 port log pid k body resp=''
  port=$((20000 + RANDOM % 20000))
  log=${TMPDIR:-/tmp}/depth-glm5next-server.$$.log
  "${CMD[@]}" --port "$port" > "$log" 2>&1 &
  pid=$!
  rc=124
  for ((k = 0; k < BOUND / 2; k++)); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/health"; then rc=0; break; fi
    if ! kill -0 "$pid" 2> /dev/null; then
      wait "$pid"
      rc=$?
      [ "$rc" != 0 ] || rc=70
      break
    fi
    sleep 2
  done
  if [ "$rc" = 0 ]; then
    body=$(prompt_ids "$dep" | python3 -c '
import json, sys
ids = [int(l) for l in sys.stdin if l.strip()]
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[1]), "temperature": 0, "ignore_eos": True, "cache_prompt": False}))' "$N")
    resp=$(curl -sf --max-time "$BOUND" -H 'Content-Type: application/json' --data-binary @- "http://127.0.0.1:$port/completion" <<< "$body") || rc=$?
  fi
  # TERM to the timeout, which passes it to the server; a server still up 30 s later is killed by its
  # parent's pid, the one started here, never by a name.
  kill "$pid" 2> /dev/null
  for ((k = 0; k < 30; k++)); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
  if kill -0 "$pid" 2> /dev/null; then
    echo "[server] still up 30 s after TERM; KILL to the children of $pid, then $pid" >&2
    pkill -KILL -P "$pid"
    kill -KILL "$pid" 2> /dev/null
  fi
  wait "$pid" 2> /dev/null
  out="$(cat "$log")
RESPONSE $resp"
  rm -f "$log"
}

# run_arm <tag> <round> <index>: tag is ROW (a round's arm), WARMUP or DISCARD (discarded rows).
run_arm() {
  local tag=$1 r=$2 i=$3 eng=${A_ENG[$3]} dep=${A_DEP[$3]} label=${A_LABEL[$3]} out rc t0 t1 m0 m1 val w rowtags why line markf mark
  CPU_BUSY_TAG='' FIT_COL='' FIT_LINES=''
  guard_other
  guard_timing
  preheat_arm "$i" || {
    rc=$?
    [ "$tag" != ROW ] || n_rows=$((n_rows + 1))
    if [ "$tag" = ROW ]; then fail_row "" "$r" "$i" "$rc" "$FAIL_WHY" ""; else fail_row "" 0 "$i" "$rc" "$FAIL_WHY" ""; fi
    return
  }
  guard_cpu "pre r$r $label $dep"
  arm_cmd "$i"
  witness "pre $tag r$r $label ${dep}"
  ref_witness
  m0=$(majflt) t0=$(date +%s)
  case $eng in
    lcpp2775[24]srv | lcpp2775[24]mtp) srv_run "$dep" ;;
    ours | hot)
      # Through majflt_mark: the fault count at the `fed` line, where the feed's timer starts.
      markf=$(mktemp "${TMPDIR:-/tmp}/depth-glm5next-fed.XXXXXX") || exit 2
      out=$(lease_bounded "$BOUND" "${CMD[@]}" 2>&1 | majflt_mark "$markf" '^fed '; exit "${PIPESTATUS[0]}")
      rc=$?
      mark=$(cat "$markf")
      rm -f "$markf"
      ;;
    *)
      out=$(lease_bounded "$BOUND" "${CMD[@]}" 2>&1)
      rc=$?
      ;;
  esac
  t1=$(date +%s) m1=$(majflt)
  witness "post $tag r$r $label ${dep}"
  guard_cpu "post r$r $label $dep"
  [ "$tag" != ROW ] || {
    n_rows=$((n_rows + 1))
    [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
    [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
  }
  local r_tag=r$r
  [ "$tag" = ROW ] || r_tag=r0
  # A fit arm's loader lines say what the fit chose; a fit that failed or never ran fails the arm by
  # name, whatever llama-bench measured or how it exited after it.
  if fit_eng "$eng" && ! lcpp_fit_col "$out"; then
    why=$FIT_WHY
    [ "$rc" = 0 ] || why="exited $rc; $why"
    fail_row "" "${r_tag#r}" "$i" "$rc" "$why" "$out"
    return
  fi
  if [ "$rc" -ne 0 ]; then fail_row "" "${r_tag#r}" "$i" "$rc" "exited $rc" "$out"; return; fi
  case $eng in
    ours | hot)
      local rec P50 MEAN WARMCOL PLACE_RAN SERIES TOKENS PP_N PP_MS PP_TPS PP_PASSES PP_KIND CARD_EXP HOST_EXP
      rec=$(python3 "$RECORDS" sh --bin generate_glm5next - P50=smoke.p50_ms MEAN=smoke.mean_ms WARMCOL=smoke.warm \
        PLACE_RAN=smoke.place 'SERIES=time_step.ms*' 'TOKENS=step.token*' PP_N=time_prompt.n PP_MS=time_prompt.ms \
        'PP_TPS=time_prompt.tok/s' PP_PASSES=time_prompt.passes PP_KIND=time_prompt.kind CARD_EXP=plan.card_experts \
        HOST_EXP=plan.host_experts <<< "$out") || { fail_row "" "${r_tag#r}" "$i" 0 "records.py did not read the output" "$out"; return; }
      eval "$rec"
      [ -n "$P50" ] && [ -n "$PP_N" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no SMOKE or time prompt record" "$out"; return; }
      [ "$PLACE_RAN" = "$PLACE" ] || { fail_row "" "${r_tag#r}" "$i" 0 "its SMOKE names place=$PLACE_RAN; the runner passed --place $PLACE" "$out"; return; }
      [ "$eng" != hot ] || [ "${CARD_EXP:-0}" -gt 0 ] || { fail_row "" "${r_tag#r}" "$i" 0 "plan card_experts=${CARD_EXP:-?} under BLOOMERY_HOT_LIST: this binary keeps every routed expert on the host" "$out"; return; }
      echo "$out" | grep -E '^(plan|load|capture|fed|step 0|time prompt) '
      local h10 t10 uniq tps tps50
      h10=$(echo "$SERIES" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
      t10=$(echo "$SERIES" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
      uniq=$(echo "$TOKENS" | sort -u | grep -c .)
      tps=$(awk -v m="$MEAN" 'BEGIN{printf "%.2f", 1e3/m}')
      tps50=$(awk -v p="$P50" 'BEGIN{printf "%.2f", 1e3/p}')
      w=$(awk -v a="$PP_MS" -v m="$MEAN" -v n="$N" 'BEGIN{print (a + (n - 1) * m) / 1e3}')
      if [ -n "$mark" ]; then
        cold_col "$((m1 - mark))" "$w" "timed, from the fed line; whole process $((m1 - m0))"
      else
        cold_col "$((m1 - m0))" "$w" "whole process: no fed line"
      fi
      cold_pass "$tag" "$r" "$i"
      prompt_col "$eng" "$dep"
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG$TRACE_TAG"
      echo "$PTAG $r_tag $label d=$dep n=$N ctx=$CTX | tok/s(mean) $tps @ n=$N, depth $dep, $CARD_NAME | place $PLACE_RAN card_experts $CARD_EXP host_experts $HOST_EXP$PROMPT_COL | p50 $P50 ms | mean $MEAN ms | tok/s(p50) $tps50 | warm ${WARMCOL:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq | pp_tok/s $PP_TPS (n=$PP_N, passes=$PP_PASSES, kind=$PP_KIND)$MAJ_COL | wall $((t1 - t0))s$rowtags"
      [ "$PTAG" = ROW ] || { cold_count "$tag"; return 0; }
      sums+=("$label|$dep|$r|$tps")
      pp_sums+=("$label|$PP_N|$r|$PP_TPS")
      ;;
    lcpp2775[24]srv | lcpp2775[24]mtp)
      local t acc pn prn prps dn dna
      t=$(sed -n 's/^RESPONSE //p' <<< "$out" | python3 -c '
import json, sys
t = json.loads(sys.stdin.read())["timings"]
print(t["predicted_per_second"], t["predicted_n"], t["prompt_n"], t["prompt_per_second"], t.get("draft_n", 0), t.get("draft_n_accepted", 0))' 2> /dev/null) ||
        { fail_row "" "${r_tag#r}" "$i" 0 "no timings in the /completion response" "$out"; return; }
      read -r val pn prn prps dn dna <<< "$t"
      [ "$pn" = "$N" ] || { fail_row "" "${r_tag#r}" "$i" 0 "predicted_n $pn, not $N" "$out"; return; }
      acc=$(grep -o 'draft acceptance = .*' <<< "$out" | tail -n 1)
      case $eng in *mtp) [ "$dn" -gt 0 ] || { fail_row "" "${r_tag#r}" "$i" 0 "the MTP arm drafted nothing (draft_n 0)" "$out"; return; } ;; esac
      echo "$out" | grep -v '^RESPONSE ' | grep -E '^build:|model buffer size|speculative|draft-mtp|nextn' | head -n 8 | sed "s/^/    $label load /"
      val=$(awk -v v="$val" 'BEGIN{printf "%.2f", v}')
      cold_col "$((m1 - m0))" "$(awk -v n="$N" -v v="$val" 'BEGIN{print n / v}')"
      cold_pass "$tag" "$r" "$i"
      prompt_col "$eng" "$dep"
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
      echo "$PTAG $r_tag $label d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | llama-server /completion$PROMPT_COL | prompt_n $prn prompt tok/s $(awk -v v="$prps" 'BEGIN{printf "%.2f", v}') | draft_n $dn draft_n_accepted $dna | ${acc:-no draft acceptance line}$MAJ_COL | wall $((t1 - t0))s$rowtags"
      [ "$PTAG" = ROW ] && sums+=("$label|$dep|$r|$val")
      ;;
    lcpp*)
      val=$(echo "$out" | grep -F "$LABEL_TEST" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//; s/ //g' | head -n 1)
      [ -n "$val" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no '${LABEL_TEST% |}' row" "$out"; return; }
      echo "$out" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $label table /"
      [ -z "$FIT_COL" ] || while IFS= read -r line; do echo "    $label fit $line"; done <<< "$FIT_LINES"
      local build dev
      build=$(echo "$out" | sed -n 's/^build: //p' | head -n 1)
      dev=$(echo "$out" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
      if [ "${A_PP[$i]}" = 1 ]; then
        cold_col "$((m1 - m0))" "$(awk -v p="$dep" -v v="$val" 'BEGIN{print p / v}')"
        cold_pass "$tag" "$r" "$i"
        rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
        echo "$PTAG $r_tag $label p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME$FIT_COL | $BATCH | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] && pp_sums+=("$label|$dep|$r|$val")
      else
        cold_col "$((m1 - m0))" "$(awk -v n="$N" -v v="$val" 'BEGIN{print n / v}')"
        cold_pass "$tag" "$r" "$i"
        rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
        echo "$PTAG $r_tag $label d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME$FIT_COL | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] && sums+=("$label|$dep|$r|$val")
      fi
      ;;
    exl3*)
      local clean
      clean=$(echo "$out" | strip)
      val=$(echo "$clean" | grep -E "^${LABEL_TEST%%[0-9]*} +$dep:" | grep -oE '[0-9]+\.[0-9]+ +tokens/s' | head -n 1 | sed 's/ .*//')
      [ -n "$val" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no '$LABEL_TEST' row" "$out"; return; }
      echo "$clean" | grep -E '^ -- (Bitrate|Chunk size)|CPU MoE worker started' | sed "s/^/    $label load /"
      echo "    $label load CPU split: $(echo "$clean" | grep -c 'CPU split experts') layers, first: $(echo "$clean" | grep -m1 'CPU split experts' | sed 's/.*mlp //')"
      echo "$clean" | grep -E '^(Context|Length) +[0-9]+:' | sed "s/^/    $label table /"
      MAJ_COL=" | majflt $((m1 - m0)) (whole process, its load's; untagged)" COLD_TAG='' PTAG=$tag
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG"
      if [ "${A_PP[$i]}" = 1 ]; then
        echo "$tag $r_tag $label p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME | EXL3 $EXL3_BPW bpw, another quantization | chunk 4096$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$tag" = ROW ] && pp_sums+=("$label|$dep|$r|$val")
      else
        echo "$tag $r_tag $label d=$dep n=100 | tok/s $val @ n=100, depth $dep, $CARD_NAME | EXL3 $EXL3_BPW bpw, another quantization$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$tag" = ROW ] && sums+=("$label|$dep|$r|$val")
      fi
      ;;
  esac
  cold_count "$tag"
  return 0
}
# cold_count <tag>: a round's [cold] row into cold_rows.
cold_count() { [ "$1" != ROW ] || [ -z "$COLD_TAG" ] || cold_rows=$((cold_rows + 1)); }
# run_row <round> <index>: a round's row of arm <index> under the cold rule (the header): a COLD row runs
# the arm once more.
run_row() {
  COLD_PASS=0 COLD_AGAIN=0
  run_arm ROW "$1" "$2"
  [ "$COLD_AGAIN" = 1 ] || return 0
  retry_rows=$((retry_rows + 1))
  echo "[cold] r$1 ${ARMS[$2]} read [cold]: its arm runs once more (docs/fair-measure.md 2.3)"
  COLD_PASS=1 COLD_AGAIN=0
  run_arm ROW "$1" "$2"
  COLD_PASS=0
}

# ratio_table <prefix> <keys> <labels>: records `label|key|round|value` on stdin; for every key and
# every label, each round's ours / label ratio, their mean with its 95 % interval (Student t at rounds
# - 1 degrees of freedom) and the ratio of the arm means (depth-qwen3moe.sh's table).
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
} END {
  nt = split(t975, t, " "); nd = split(deps, d, " "); nr = split(refs, rf, " ")
  for (i = 1; i <= nd; i++) for (j = 1; j <= nr; j++) {
    ref = rf[j]
    if (!(("ours" SUBSEP d[i]) in an) || !((ref SUBSEP d[i]) in an)) continue
    c = 0; m = 0; list = ""
    for (r = 1; r <= rounds; r++) {
      ko = "ours" SUBSEP d[i] SUBSEP r; kr = ref SUBSEP d[i] SUBSEP r
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
    printf "%s%-5s ours/%-16s mean %.4f %s (n=%d)  of means %.4f  per round:%s\n", prefix, d[i], ref, m, ci, c, (as["ours" SUBSEP d[i]] / an["ours" SUBSEP d[i]]) / (as[ref SUBSEP d[i]] / an[ref SUBSEP d[i]]), list
  }
}'
}
means() { # means <unit>: `label|key|round|value` on stdin, one mean line per label and key
  awk -F'|' -v unit="$1" '{
    k = $1 " " $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
  } END { for (k in s) printf "mean %-26s %9.2f %s  [%s..%s, spread %.2f%%]  (n=%d)\n", k, s[k] / n[k], unit, mn[k], mx[k], (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0, n[k] }' | sort
}

majflt_require depth-glm5next.sh
lease_take
echo "[config] model=$MODEL n=$N rounds=$ROUNDS card=$CARD_NAME timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU arm_bound=${BOUND}s cold_us=$COLD_US"
[ "$ours" = 0 ] || echo "[config] ours: $BIN --ctx $CTX --place $PLACE warm=${WARM:-0} prose=$PROSE hot=$HOT"
[ "$ours$srv" = 00 ] || echo "[config] prompt: GLM_PROSE ids from index $PROSE_FROM; the hot list's trace: ids 0..$((GLM_HOT_TRACE - 1))"
[ "$lcpp" = 0 ] || echo "[config] lcpp27752 flags=$LCPP27752_GPU_FLAGS | lcpp27754 env=$LCPP27754_ENV flags=$LCPP27754_GPU_FLAGS"
for e in 27752 27754; do
  case $e in 27752) [ "$fit27752" = 1 ] || continue ;; *) [ "$fit27754" = 1 ] || continue ;; esac
  lcpp_fit_flags "$(gpu_flags "lcpp$e")"
  echo "[config] lcpp${e}fit: flags=$FIT_FLAGS (llama-bench's fit places the model; dropped: $FIT_DROPPED; lcpp${e}ppfit<U>: -ub U -b max(U, 2048))"
done
[ "$exl3" = 0 ] || echo "[config] exl3: $EXL3_MODEL ($EXL3_BPW bpw) flags=$EXL3_FLAGS, perf.py's defaults otherwise (cache 32768, chunk 4096)"
echo "[config] arms=${ARMS[*]}"
if [ -n "$PH_DIR" ]; then
  for var in ${!PH_LINE_*}; do echo "[config] preheat: ${!var}"; done
else
  echo "[config] preheat: off ($([ "$PREHEAT" = 0 ] && echo BLOOMERY_PREHEAT=0 || echo 'no GGUF arm'))"
fi
witness pre
ref_witness
for grp in 0 1; do
  case $grp in 0) [ "$gguf" = 1 ] || continue ;; 1) [ "$exl3" = 1 ] || continue ;; esac
  [ "$WARMUP" = 0 ] || [ "$grp" = 0 ] || run_arm DISCARD 0 "$(first_of 1)"
  for r in $(seq "$ROUNDS"); do
    for i in $(order_of "$r" "$grp"); do
      # Each GGUF arm's warm-up: the same ids, discarded, right before its first round's row.
      [ "$WARMUP" = 0 ] || [ "$grp" = 1 ] || [ "$r" != 1 ] || run_arm WARMUP 0 "$i"
      run_row "$r" "$i"
    done
  done
done

echo
echo "arm runs: $n_rows (FAIL rows included); rows tagged [cpu-busy] $busy_rows, [other-busy] $other_rows, [cold] $cold_rows; cold re-runs $retry_rows, FAIL-cold $fail_cold"
echo "=== per-arm decode means (tok/s @ n=$N, $CARD_NAME; exl3 rows @ n=100, another quantization) ==="
[ ${#sums[@]} -eq 0 ] || printf '%s\n' "${sums[@]}" | means tok/s
echo "=== per-arm prefill means (tok/s(pp) @ n=0, prompt P, $CARD_NAME; ours is the step feed, kind=steps) ==="
[ ${#pp_sums[@]} -eq 0 ] || printf '%s\n' "${pp_sums[@]}" | means 'tok/s(pp)'
# Ratios against ours: the same file's engines only. exl3 runs another quantization. The engines do not
# share a routing condition: ours feeds prose ids and places card experts by id or by a prose hot
# list; llama-bench feeds std::rand() ids and places whole layers, whose host bytes a token do not
# depend on the ids; exllamav3 feeds wikitext-2 and adapts its placement to that stream.
same_file() { grep -vE '^exl3' | grep -vx ours; }
if [ ${#sums[@]} -gt 0 ]; then
  echo
  echo "=== ours / each engine on the same file, per depth: each round's ratio, their mean ± 95 % (t, rounds - 1 df) ==="
  keys=$(printf '%s\n' "${sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  refs=$(printf '%s\n' "${sums[@]}" | cut -d'|' -f1 | sort -u | same_file | tr '\n' ' ')
  printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$keys" "$refs"
fi
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== ours / each engine on the same file, prefill per prompt length (ours: the step feed) ==="
  keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | sort -u | same_file | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$keys" "$refs"
fi
witness post
ref_witness
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed arms: ${failed[*]}"
  exit 1
fi
