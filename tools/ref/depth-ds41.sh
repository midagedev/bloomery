#!/usr/bin/env bash
# V4.1 decode by depth, both engines in one lease on the timing card (run on the box, lead-only):
# our engine (generate_ds41 --depth D --time, placement (a)) and ik (llama-bench -gp D,N at the
# profile's IK_GPU_FLAGS, under its IK_GPU_ENV), alternated arm by arm.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/depth-ds41.sh 6 ik:6'
#   just depth-gpu-ds41 6 ik:6 1024 ik:1024 4096 ik:4096
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
#   ik:<D>   ik: llama-bench -p 0 -n 0 -gp D,N -r 1 (D = 0: plain tg N). llama-bench sizes its
#            context to D + N itself and feeds its own prompt ids, not lcg_prompt's. It runs as
#            `env $IK_GPU_ENV`: without GGML_CUDA_NO_PINNED_WEIGHTS the CPU expert overrides turn
#            the host context into one pinned allocation larger than RAM and the load fails (the
#            profile's IK_GPU_ENV comment).
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
#
# Paging. The two engines' host expert sets differ (ours: experts n_l.. of every layer, ik: all
# experts of its CPU layers), and together they are about the page cache's size. The witness
# prints the major fault count and the page cache before and after every arm, so an arm that paged
# its set back in shows. BLOOMERY_GEN_WARM (generate_ds41 --warm) trims our arm's first steps.
# With BLOOMERY_STEP_STATS=1 (BLOOMERY_BOX_ENV on the Mac side) our arm's `stat summary` line is
# echoed with its load lines; the per-step `stat step` lines stay in the arm's output only.
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 3),
# BLOOMERY_GEN_WARM, BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_ARM_BOUND
# (seconds one arm may run, default 900: a hung arm fails the runner with rc 124/137 instead of
# holding the lease).
set -uo pipefail
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_ENV); tools/box.sh exports its MODEL to our
# binary as BLOOMERY_REF_MODEL, so both engines open one file.
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
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6)
for a in "${ARMS[@]}"; do
  case $a in
    ik:[0-9]* | [0-9]*) case ${a#ik:} in *[!0-9]*) echo "depth-ds41.sh: arm '$a' is <D> or ik:<D>" >&2; exit 64 ;; esac ;;
    *) echo "depth-ds41.sh: arm '$a' is <D> or ik:<D>" >&2; exit 64 ;;
  esac
done
# The card pin, the card's witness lines, the other-card guard and the binary's freshness.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# Our binary matters only to our arms: an ik-only run neither builds it (the recipe skips the build)
# nor reads it, so its freshness is not asked.
ours=0
for a in "${ARMS[@]}"; do case $a in ik:*) ;; *) ours=1 ;; esac; done
if [ "$ours" = 1 ]; then assert_fresh_binary "$BIN" || exit $?; fi
[ -x "$IKBIN" ] || { echo "depth-ds41.sh: no llama-bench at $IKBIN" >&2; exit 2; }
CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
IK_SHA=$(sha256sum "$IKBIN" | cut -c1-12)
IK_HEAD=$(git -c safe.directory="$IK" -C "$IK" rev-parse --short=8 HEAD 2> /dev/null || echo '?')
IK_DIRTY=$(git -c safe.directory="$IK" -C "$IK" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')

# The witness before and after every row: the timing card's lines (with our binary), the busiest
# processes, the page cache and the major fault count.
WITNESS=(head-open indent card busiest model mem pgmajfault)
ik_witness() { echo "    ik: $IKBIN sha256=$IK_SHA head=$IK_HEAD dirty_files=$IK_DIRTY"; }

lease_take
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] ours: $BIN (placement (a), default ctx) ik: $IKBIN flags=$IK_GPU_FLAGS env=$IK_GPU_ENV"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
ik_witness
guard_other

sums=()
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#ARMS[@]} - 1))); do
    a=${ARMS[$(((i + r - 1) % ${#ARMS[@]}))]}
    guard_other
    case $a in
      ik:*)
        dep=${a#ik:}
        witness "pre r$r ik d=$dep"
        ik_witness
        t0=$(date +%s)
        if [ "$dep" = 0 ]; then
          # shellcheck disable=SC2086
          raw=$(timeout --kill-after=10 "$BOUND" env $IK_GPU_ENV "$IKBIN" -m "$MODEL" -p 0 -n "$N" -r 1 $IK_GPU_FLAGS 2>&1)
          rc=$?
          label="tg$N "
        else
          # shellcheck disable=SC2086
          raw=$(timeout --kill-after=10 "$BOUND" env $IK_GPU_ENV "$IKBIN" -m "$MODEL" -p 0 -n 0 -gp "$dep,$N" -r 1 $IK_GPU_FLAGS 2>&1)
          rc=$?
          label="tg$N@pp$dep"
        fi
        val=$(echo "$raw" | grep -F "$label" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
        t1=$(date +%s)
        witness "post r$r ik d=$dep"
        if [ $rc -ne 0 ] || [ -z "$val" ]; then
          # The whole output goes to a file: the loader's reason for a failed load is many lines
          # above the tail.
          fail=${TMPDIR:-/tmp}/depth-ds41-ik-d${dep}-r${r}.log
          echo "$raw" > "$fail"
          echo "r$r ik d=$dep produced no tg line (rc $rc); full output: $fail" >&2
          echo "$raw" | tail -n 20 >&2
          exit 1
        fi
        echo "ROW r$r ik d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | wall $((t1 - t0))s"
        sums+=("ik d=$dep n=$N|$val|")
        ;;
      *)
        dep=$a
        witness "pre r$r ours d=$dep n=$N"
        t0=$(date +%s)
        out=$(timeout --kill-after=10 "$BOUND" "$BIN" --depth "$dep" -n "$N" --time ${WARM:+--warm "$WARM"} 2>&1)
        rc=$?
        t1=$(date +%s)
        witness "post r$r ours d=$dep n=$N"
        if [ $rc -ne 0 ]; then
          echo "r$r ours d=$dep FAILED rc=$rc" >&2
          echo "$out" | tail -n 20 >&2
          exit 1
        fi
        smoke=$(echo "$out" | grep -E '^SMOKE ')
        [ -n "$smoke" ] || { echo "r$r ours d=$dep produced no SMOKE line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
        echo "$out" | grep -E '^(plan|load|capture|fed|stat summary) '
        p50=$(echo "$smoke" | sed 's/.*p50_ms=\([0-9.]*\).*/\1/')
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
        echo "ROW r$r ours d=$dep n=$N | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok${draft:+ | draft $draft} | wall $((t1 - t0))s"
        if [ -n "$draft" ]; then tps_mean=${draft##*tok/s(positions)=}; fi
        sums+=("ours d=$dep n=$N|$tps_mean|$tps_p50")
        ;;
    esac
  done
done
echo
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, ik llama-bench's"
echo "    own mean — the cross-engine ratio reads these. The p50 column is ours only. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  s[$1]+=$2; n[$1]++; if($3!=""){sp[$1]+=$3; np[$1]++}
  if(mn[$1]==""||$2+0<mn[$1]+0)mn[$1]=$2; if(mx[$1]==""||$2+0>mx[$1]+0)mx[$1]=$2
} END{for(k in s){
  spread = (mn[k]>0) ? 100*(mx[k]-mn[k])/mn[k] : 0
  printf "mean %-18s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)\n", k, s[k]/n[k], mn[k], mx[k], spread, (np[k]?sprintf("%.2f tok/s(p50)", sp[k]/np[k]):""), n[k]}}' | sort
witness post
ik_witness
