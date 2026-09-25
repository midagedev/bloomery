#!/usr/bin/env bash
# V4.1 decode by depth, three engines in one lease on the timing card (run on the box, lead-only):
# our engine (generate_ds41 --depth D --time, placement (a)), ik (llama-bench -gp D,N at the
# profile's IK_GPU_FLAGS, under its IK_GPU_ENV) and mainline llama.cpp (llama-bench -d D at the
# profile's LCPP_GPU_FLAGS), alternated arm by arm.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/depth-ds41.sh 6 ik:6 lcpp:6'
#   just depth-gpu-ds41 6 lcpp:6 4096 lcpp:4096
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
#   <D>@NAME=VALUE[,NAME=VALUE...]  ours at depth D with those variables set (`env NAME=VALUE ...`):
#            a lever arm of the same binary, row label `ours@NAME=VALUE[,...]`. Beside a plain `<D>`
#            arm it is the same-binary A/B, e.g. `6 6@BLOOMERY_LAUNCH_THREAD=1`.
#            An arm with BLOOMERY_DRAFT=dspark also sees the other card, where the draft runs, and
#            gets the profile's DSPARK_MODEL unless it names one (timing-card.sh dspark_env).
#   bin:<path>:<D>  a second generate_ds41 (an absolute path on the box, a base tree's build) at depth
#            D, row label `bin:<basename of its tree>` (the tree is the path above `target/`). It is a
#            base by construction, so its freshness is not asked; its tree line (sha256, HEAD, dirty
#            files) is printed with the references'.
# Every label is its own engine in the per-arm means and in the ratio table (ours / each other label).
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
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 3),
# BLOOMERY_GEN_WARM, BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_ARM_BOUND
# (seconds one arm may run, default 900: a hung arm fails the runner with rc 124/137 instead of
# holding the lease), BLOOMERY_DRY=1 (print each arm's command line, the binaries' tree lines, the
# CPU guard's settings and reading, and the rotation, then exit 0 before the lease: nothing is loaded
# and nothing is timed).
#
# Contention. Before every arm the runner checks both the other card (guard_other, timing-card.sh:
# `[other-busy]`) and the CPU (guard_cpu, lease.sh: `[cpu-busy]` when builds or reference engines it
# did not start — BLOOMERY_CPU_BUSY_COMMS — sum past BLOOMERY_CPU_BUSY_PCT percent of one cpu), and
# checks the CPU again after the arm; a row that met CPU contention ends in ` [cpu-busy]` and the
# closing summary counts those rows. BLOOMERY_OTHER_STRICT=1 aborts (rc 75) on either instead.
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
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6)
ours=0 ik=0 lcpp=0
# Per arm, by its index in ARMS: the kind (ours, ref or bin), the depth, the row label, the
# reference engine (ref), the binary (ours and bin) and the NAME=VALUE list (comma-separated).
A_KIND=() A_DEP=() A_LABEL=() A_ENG=() A_BIN=() A_ENV=()
arm_usage() {
  echo "depth-ds41.sh: arm '$1' is <D>, <D>@NAME=VALUE[,NAME=VALUE...], ik:<D>, lcpp:<D>, lcpp<K>:<D> or bin:<path>:<D>" >&2
  exit 64
}
for a in "${ARMS[@]}"; do
  kind=ref eng=${a%%:*} dep=${a#*:} label='' bin='' envs=''
  case $a in
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
      IFS=, read -r -a kv <<< "$envs"
      [ ${#kv[@]} -gt 0 ] || arm_usage "$a"
      for e in "${kv[@]}"; do
        [[ $e =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:],]+$ ]] || arm_usage "$a"
      done
      ours=1
      ;;
    *:*)
      case $eng in
        ik) ik=1 ;;
        lcpp | lcpp[0-9] | lcpp[0-9][0-9]) lcpp=1 ;;
        *) arm_usage "$a" ;;
      esac
      label=$eng
      ;;
    *) kind=ours eng=ours dep=$a bin=$BIN label=ours ours=1 ;;
  esac
  case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
  A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=("$bin") A_ENV+=("$envs")
done
# The card pin, the card's witness lines, the other-card guard and the binary's freshness.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
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

# ref_cmd <engine> <depth>: the reference arm's binary, arguments and row label, into REF_ENV,
# REF_BIN, REF_ARGS and REF_LABEL. The flags are word-split on purpose: the profile keeps them as
# one string. An lcpp<K> engine is LCPP_GPU_FLAGS with its --n-cpu-moe value replaced by K.
ref_cmd() {
  local eng=$1 dep=$2 flags
  REF_ENV=()
  if [ "$eng" = ik ]; then
    # shellcheck disable=SC2206
    REF_ENV=($IK_GPU_ENV)
    REF_BIN=$IKBIN flags=$IK_GPU_FLAGS
    if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n 0 -gp "$dep,$N"); REF_LABEL="tg$N@pp$dep |"; fi
  else
    REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS
    if [ "$eng" != lcpp ]; then
      flags=$(echo " $flags " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
      flags="${flags# }--n-cpu-moe ${eng#lcpp}"
    fi
    if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n "$N" -d "$dep"); REF_LABEL="tg$N @ d$dep |"; fi
  fi
  # shellcheck disable=SC2206
  REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
}

# One reference arm: its llama-bench on its flags, the row, and the sum.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 raw rc val build dev t0 t1 fail
  ref_cmd "$eng" "$dep"
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  raw=$(timeout --kill-after=10 "$BOUND" env "${REF_ENV[@]}" "$REF_BIN" "${REF_ARGS[@]}" 2>&1)
  rc=$?
  t1=$(date +%s)
  witness "post r$r $eng d=$dep"
  guard_cpu "post r$r $eng d=$dep"
  val=$(echo "$raw" | grep -F "$REF_LABEL" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
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
  echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
  build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
  dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
  echo "ROW r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$CPU_BUSY_TAG"
  [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
  sums+=("$eng|$dep|$r|$val|")
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
# One arm of a generate_ds41: ours (our binary, with the arm's variables when it has any) or a
# second binary. The row and the sum under the arm's label.
# ours_arm <index> <round>
ours_arm() {
  local i=$1 r=$2 dep label bin out rc t0 t1 smoke p50 mean warmcol series draft h10 t10 uniq_tok tps_mean tps_p50
  local -a envs=()
  dep=${A_DEP[$i]} label=${A_LABEL[$i]} bin=${A_BIN[$i]}
  arm_envs "$i"
  envs=("${ARM_ENVS[@]}")
  witness "pre r$r $label d=$dep n=$N"
  t0=$(date +%s)
  if [ ${#envs[@]} -eq 0 ]; then
    out=$(timeout --kill-after=10 "$BOUND" "$bin" --depth "$dep" -n "$N" --time ${WARM:+--warm "$WARM"} 2>&1)
  else
    out=$(timeout --kill-after=10 "$BOUND" env "${envs[@]}" "$bin" --depth "$dep" -n "$N" --time ${WARM:+--warm "$WARM"} 2>&1)
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
  echo "ROW r$r $label d=$dep n=$N | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok${draft:+ | draft $draft} | wall $((t1 - t0))s$CPU_BUSY_TAG"
  [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
  if [ -n "$draft" ]; then tps_mean=${draft##*tok/s(positions)=}; fi
  sums+=("$label|$dep|$r|$tps_mean|$tps_p50")
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
        echo "[dry] $a: timeout --kill-after=10 $BOUND env ${REF_ENV[*]} $REF_BIN ${REF_ARGS[*]}   # row label '${REF_LABEL% |}'"
        ;;
      *)
        note=''
        [ "${A_LABEL[$i]}" = ours ] || note="   # row label '${A_LABEL[$i]}'"
        arm_envs "$i"
        echo "[dry] $a: timeout --kill-after=10 $BOUND ${ARM_ENVS[*]:+env ${ARM_ENVS[*]} }${A_BIN[$i]} --depth $dep -n $N --time${WARM:+ --warm $WARM}$note"
        ;;
    esac
  done
  for r in $(seq "$ROUNDS"); do
    order=()
    for i in $(seq 0 $((${#ARMS[@]} - 1))); do order+=("${ARMS[$(((i + r - 1) % ${#ARMS[@]}))]}"); done
    echo "[dry] round $r order: ${order[*]}"
  done
  exit 0
fi

lease_take
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] ours: $BIN (placement (a), default ctx)"
echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS env=$IK_GPU_ENV"
echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS (lcpp<K>: --n-cpu-moe K)"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0}"
witness pre
ref_witness
guard_other
guard_cpu pre

sums=()
busy_rows=0
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
echo "cpu-busy rows: $busy_rows of ${#sums[@]} (BLOOMERY_CPU_BUSY_PCT=${CPU_BUSY_PCT}% over [$CPU_BUSY_COMMS])"
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
refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -vx ours | sort -u | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | awk -F'|' -v deps="$deps" -v refs="$refs" -v rounds="$ROUNDS" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
} END {
  split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086", t, " ")
  nd = split(deps, d, " ")
  nr = split(refs, rf, " ")
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
    ci = (c > 1) ? sprintf("± %.4f", ((c - 1 <= 20) ? t[c - 1] : 2.0) * sqrt(ss / (c - 1)) / sqrt(c)) : "(one round: no interval)"
    printf "ratio d=%-5s ours/%-6s  mean %.4f %s (n=%d)  of means %.4f  per round:%s\n", d[i], ref, m, ci, c, (as["ours" SUBSEP d[i]] / an["ours" SUBSEP d[i]]) / (as[ref SUBSEP d[i]] / an[ref SUBSEP d[i]]), list
  }
}'
witness post
ref_witness
