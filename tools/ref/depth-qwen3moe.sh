#!/usr/bin/env bash
# Qwen3-30B-A3B decode by depth with the whole model on the timing card, three engines in one lease
# (run on the box, lead-only): our engine (generate_qwen3moe --time), ik (llama-bench -gp D,N) and
# mainline llama.cpp (llama-bench -d D), alternated arm by arm.
#
#   BLOOMERY_MODEL=qwen3moe tools/box.sh 'bash tools/ref/depth-qwen3moe.sh 6 ik:6 lcpp:6'
#   just depth-gpu-qwen3moe 6 ik:6 ikdef:6 lcpp:6 1024 ik:1024 ikdef:1024 lcpp:1024 4096 ik:4096 ikdef:4096 lcpp:4096
#
# depth-ds41.sh's shape, and it blocks the same failure: a ratio read at one depth and quoted as
# "decode is faster" — a step's attention term grows with the cached keys, so the depth goes into
# every number (`tok/s @ n=N, depth D, <card>`) and the ratio table at the end is per depth.
#
# Arms, in lease order (the order rotates by one slot each round — the position bias ab-decode.sh
# names):
#   <D>       ours, D >= 1: generate_qwen3moe --tokens <lcg_prompt D> -n N --ctx C --time. The D
#             fed ids are D real decode steps, untimed (lease.sh's lcg_prompt: 100000, then ids in
#             [1000, 91000), all inside this vocabulary); generated token 0 comes out of the last
#             of them and the N - 1 steps after it are timed. C is D + N rounded up to 256: both
#             llama-benches size n_ctx to D + N for this test and pad it to 256 under flash
#             attention (ik: GGML_PAD(n_ctx, llama_kv_cache::get_padding(flash_attn)) in
#             src/llama.cpp; mainline: GGML_PAD(n_ctx, 256) in src/llama-context.cpp), so the three
#             caches have one height.
#             Our flash grid is fixed by C (flash_gqa::segments_for), so C goes into the row.
#             BLOOMERY_GEN_CTX fixes C for every ours arm instead, e.g. at a serving height.
#   ik:<D>    ik: llama-bench -p 0 -n 0 -gp D,N -r 1 $IK_GPU_FLAGS (D = 0: plain tg N). The D-token
#             prefill runs first and untimed: ik restarts its clock after it.
#   ikdef:<D> the same at $IK_GPU_DEFAULT_FLAGS: llama-bench's own defaults with every layer on the
#             card. IK_GPU_FLAGS opts into two merges nobody has timed on this model; this arm,
#             interleaved with ik:<D>, says whether they are ik's faster set.
#   lcpp:<D>  mainline: llama-bench -p 0 -n N -d D -r 1 $LCPP_GPU_FLAGS (D = 0: plain tg N).
#             Mainline has no -gp; -d prefills D tokens before its clock starts. Its label is
#             `tgN @ dD`, ik's `tgN@ppD`.
# Both references feed std::rand() ids, prefill and steps alike; ours feeds its greedy
# continuation, which is why our row carries distinct_tokens. Every arm is one process — a model
# load, one prefill, N steps — and the three engines open the one file.
#
# Flags. All three arms hold every layer, the output head and an f16 K/V cache on the card
# (llama-bench's -ctk/-ctv default is f16; ours keeps f16 planes), so a step reads the same weight
# and cache bytes in each. The input embedding differs: both references keep it on the host and
# copy one row to the card per step (ik: buft_input in src/llama.cpp; mainline: dev_input in
# src/llama-model.cpp), ours gathers it on the card. The references' flag sets are the profile's
# (models/qwen3moe.sh), chosen by reading each tree's CLI, docs and loader; no flag sweep has been
# run, so a row is "at these flags". A sweep overrides them inside the box command, the profile
# keeps a caller's value:
#   BLOOMERY_MODEL=qwen3moe tools/box.sh 'IK_GPU_FLAGS="-ngl 99 -fa 1" bash tools/ref/depth-qwen3moe.sh ik:6'
#   ik    -ngl 99   every layer and the output head on the card (ikdef: this flag alone)
#         -fa 1     flash attention; ik's default, spelled out
#         -fmoe 1   the fused MoE up-gate op; ik's default, spelled out
#         -mqkv 1   attn_q/k/v loaded as one matrix where their types agree (the 24 layers whose
#                   attn_v is Q4_K; the 24 with a Q6_K attn_v merge q and k): fewer launches a layer
#         -muge 1   ffn_gate_exps and ffn_up_exps loaded as one matrix (Q4_K in all 48 layers)
#         With these two, ik's size and params columns count each merged matrix twice
#         (llama_model_size in src/llama.cpp sums tensors_by_name, which lists the merged matrix
#         and its views), so its size reads larger than the file's. The views allocate nothing.
#         left out: -gr (graph reuse, on by default); -rcache (build_qwen3moe never builds the
#         rope cache); -ger (BailingMoE only); -sas (split mode graph only); -mla (MLA models);
#         -cuda (fusion and graphs are on in this build; offload-batch-size* tune host offload,
#         mmq-id-size batched expert matmuls, enable-p2p multi-card copies, fa-offset the flash
#         softmax's arithmetic); -rtr, -thp, -mmp, --defer-experts, -t
#         (host tensors, load time and host threads: at -ngl 99 a step's only host work is the
#         embedding row, the input layer stays on the host); -b, -ub, -amb (the untimed prefill);
#         -ser (drops experts: another computation, not a faster one); -ctk/-ctv (a quantized
#         cache: likewise)
#   lcpp  -ngl 99   every layer and the output head on the card
#         -fa on    flash attention; mainline's default is auto
#         left out: -lm, -lzm (load mode); -nopo, --no-host (host tensors); -t, --poll, -C (host
#         threads, as above); -b, -ub (the untimed prefill); -ctk/-ctv (as above). Mainline has
#         no fused-MoE or merge flag: CUDA graphs and op fusion are build options, on in this
#         build (GGML_CUDA_GRAPHS=ON in the tree's CMakeCache.txt).
#
# The trees are the profile's IK and LCPP. The witness prints each binary's sha256 with its tree's
# HEAD and dirty count, and each reference row the binary's own `build:` line — a tree can move
# after its binary was built — and the device llama-bench opened.
#
# Paging. The witness prints the page cache and the major fault count before and after every arm:
# the file is 18.6 GB, and another round's host set can evict it between two arms.
#
# Co-tenants. A compute process on the other card is recorded ([other-busy], timing-card.sh); one
# on the timing card as an arm starts holds the arm until it is gone, and after 10 minutes stops the
# runner with rc 75 and no summary (guard_timing below).
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 4),
# BLOOMERY_GEN_WARM (generate_qwen3moe --warm), BLOOMERY_GEN_CTX (above), BLOOMERY_GEN_BIN
# (default target/release/generate_qwen3moe), BLOOMERY_ARM_BOUND (seconds one arm may run,
# default 900: a hung arm fails the runner with rc 124/137 instead of holding the lease).
set -uo pipefail
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_DEFAULT_FLAGS, LCPP, LCPPBIN, LCPP_GPU_FLAGS);
# tools/box.sh exports its MODEL to our binary as BLOOMERY_REF_MODEL, so the three engines open
# one file.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = qwen3moe ] || {
  echo "depth-qwen3moe.sh: the profile is $MODEL_NAME — pick qwen3moe on the Mac side (BLOOMERY_MODEL=qwen3moe)" >&2
  exit 64
}
N=${BLOOMERY_DECODE_N:-96}
WARM=${BLOOMERY_GEN_WARM:-}
ROUNDS=${BLOOMERY_AB_ROUNDS:-4}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_qwen3moe}
BOUND=${BLOOMERY_ARM_BOUND:-900}
GEN_CTX=${BLOOMERY_GEN_CTX:-}
case $GEN_CTX in
  *[!0-9]* | 0) echo "depth-qwen3moe.sh: BLOOMERY_GEN_CTX is a positive integer, got '$GEN_CTX'" >&2; exit 64 ;;
esac
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6 lcpp:6)
ours=0 ik=0 lcpp=0
for a in "${ARMS[@]}"; do
  case $a in
    ik:*) dep=${a#ik:}; ik=1 ;;
    ikdef:*) dep=${a#ikdef:}; ik=1 ;;
    lcpp:*) dep=${a#lcpp:}; lcpp=1 ;;
    *) dep=$a; ours=1 ;;
  esac
  case $dep in
    '' | *[!0-9]*) echo "depth-qwen3moe.sh: arm '$a' is <D>, ik:<D>, ikdef:<D> or lcpp:<D>" >&2; exit 64 ;;
  esac
  # Our prompt is one argument of D ids of up to six characters each; the kernel caps one
  # argument at 128 KiB.
  if [ "$a" = "$dep" ] && { [ "$dep" -lt 1 ] || [ "$dep" -gt 20000 ]; }; then
    echo "depth-qwen3moe.sh: our arm '$a' needs 1 <= D <= 20000 (D fed ids in one --tokens argument)" >&2
    exit 64
  fi
done
# The card pin, the card's witness lines, the other-card guard and the binary's freshness.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# Each engine's binary matters only to its own arms: a reference-only run neither builds ours (the
# recipe skips the build) nor reads it, so its freshness is not asked.
if [ "$ours" = 1 ]; then assert_fresh_binary "$BIN" || exit $?; fi
if [ "$ik" = 1 ]; then [ -x "$IKBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $IKBIN" >&2; exit 2; }; fi
if [ "$lcpp" = 1 ]; then [ -x "$LCPPBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $LCPPBIN" >&2; exit 2; }; fi
CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
# A binary's sha256 and its tree's HEAD and dirty count, once. GIT_OPTIONAL_LOCKS=0 keeps `git
# status` from rewriting the index of a tree this root process does not own.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" | cut -c1-12)
  head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=9 HEAD 2> /dev/null || echo '?')
  dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  echo "$bin sha256=$sha head=$head dirty_files=$dirty"
}
IK_LINE='' LCPP_LINE=''
[ "$ik" = 0 ] || IK_LINE=$(tree_line "$IKBIN" "$IK")
# shellcheck disable=SC2153 # LCPP is the profile's, which shellcheck does not follow
[ "$lcpp" = 0 ] || LCPP_LINE=$(tree_line "$LCPPBIN" "$LCPP")
ref_witness() {
  [ -z "$IK_LINE" ] || echo "    ik: $IK_LINE"
  [ -z "$LCPP_LINE" ] || echo "    lcpp: $LCPP_LINE"
}

# The witness before and after every row: the timing card's lines (with our binary), the busiest
# processes, the page cache and the major fault count.
WITNESS=(head-open indent card busiest model mem pgmajfault)

# A compute process on the timing card as an arm starts is another round's functional run
# (BLOOMERY_CARD=a6000 is refused while the card is busy, not while this lease is held, so one can
# start in the gap between two arms). Every arm here loads the whole model onto that card, so it
# would be timed beside that process or fail its allocation. Such a run lasts minutes: wait for it,
# polling every 10 s, and refuse after 10 minutes, rc 75 — contention, not a result.
guard_timing() {
  local apps i
  for ((i = 0; i < 60; i++)); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * 10)) s" >&2
      return 0
    fi
    if [ "$i" = 0 ]; then
      echo "[timing-busy] $(now) compute apps on the timing card ($TIMING_GPU): [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
      witness wait-timing >&2
    fi
    sleep 10
  done
  echo "[timing-busy] $(now) still busy after 10 min: [$(echo "$apps" | tr '\n' ';')]" >&2
  witness abort-timing >&2
  exit 75
}

# One reference arm: its llama-bench on its flags, the row, and the sum.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 bin flags raw rc label val build dev t0 t1 fail
  local -a cmd
  if [ "$eng" != lcpp ]; then
    bin=$IKBIN flags=$IK_GPU_FLAGS
    [ "$eng" = ik ] || flags=$IK_GPU_DEFAULT_FLAGS
    if [ "$dep" = 0 ]; then cmd=(-p 0 -n "$N"); label="tg$N |"; else cmd=(-p 0 -n 0 -gp "$dep,$N"); label="tg$N@pp$dep |"; fi
  else
    bin=$LCPPBIN flags=$LCPP_GPU_FLAGS
    if [ "$dep" = 0 ]; then cmd=(-p 0 -n "$N"); label="tg$N |"; else cmd=(-p 0 -n "$N" -d "$dep"); label="tg$N @ d$dep |"; fi
  fi
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  # shellcheck disable=SC2086
  raw=$(timeout --kill-after=10 "$BOUND" "$bin" -m "$MODEL" "${cmd[@]}" -r 1 $flags 2>&1)
  rc=$?
  t1=$(date +%s)
  witness "post r$r $eng d=$dep"
  val=$(echo "$raw" | grep -F "$label" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    # The whole output goes to a file: the loader's reason for a failed load is many lines
    # above the tail.
    fail=${TMPDIR:-/tmp}/depth-qwen3moe-$eng-d$dep-r$r.log
    echo "$raw" > "$fail"
    echo "r$r $eng d=$dep produced no '${label% |}' row (rc $rc); full output: $fail" >&2
    echo "$raw" | tail -n 20 >&2
    exit 1
  fi
  # The reference's own table, header and row: its columns name every setting it ran with that
  # differs from its defaults, so the log shows which flags took.
  echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
  build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
  dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
  echo "ROW r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s"
  sums+=("$eng|$dep|$r|$val|")
}

lease_take
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] ours: $BIN ctx=${GEN_CTX:-D+N rounded up to 256}"
echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS ikdef flags=$IK_GPU_DEFAULT_FLAGS"
echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
ref_witness
guard_other
guard_timing

sums=()
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#ARMS[@]} - 1))); do
    a=${ARMS[$(((i + r - 1) % ${#ARMS[@]}))]}
    guard_other
    guard_timing
    case $a in
      ik:*) ref_arm ik "${a#ik:}" "$r" ;;
      ikdef:*) ref_arm ikdef "${a#ikdef:}" "$r" ;;
      lcpp:*) ref_arm lcpp "${a#lcpp:}" "$r" ;;
      *)
        dep=$a
        ctx=${GEN_CTX:-$(((dep + N + 255) / 256 * 256))}
        witness "pre r$r ours d=$dep n=$N ctx=$ctx"
        t0=$(date +%s)
        out=$(timeout --kill-after=10 "$BOUND" "$BIN" --tokens "$(lcg_prompt "$dep")" -n "$N" --ctx "$ctx" --time ${WARM:+--warm "$WARM"} 2>&1)
        rc=$?
        t1=$(date +%s)
        witness "post r$r ours d=$dep n=$N ctx=$ctx"
        if [ $rc -ne 0 ]; then
          echo "r$r ours d=$dep ctx=$ctx FAILED rc=$rc" >&2
          echo "$out" | tail -n 20 >&2
          exit 1
        fi
        smoke=$(echo "$out" | grep -E '^SMOKE ')
        [ -n "$smoke" ] || { echo "r$r ours d=$dep produced no SMOKE line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
        # The prompt_ids line is the whole prompt; the load, capture and step-0 lines are the
        # arm's configuration and the fed steps' time.
        echo "$out" | grep -E '^(load|capture|step 0) '
        p50=$(echo "$smoke" | sed 's/.*p50_ms=\([0-9.]*\).*/\1/')
        mean=$(echo "$smoke" | sed 's/.*mean_ms=\([0-9.]*\).*/\1/')
        warmcol=$(echo "$smoke" | sed -n 's/.* warm=\([0-9]*\).*/\1/p')
        nodes=$(echo "$out" | sed -n 's/^capture graph_nodes=\([0-9]*\).*/\1/p')
        series=$(echo "$out" | awk '/^time step /{sub(/.*ms=/,""); print}')
        h10=$(echo "$series" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
        t10=$(echo "$series" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
        uniq_tok=$(echo "$out" | awk '/^step / && $2 != 0 {print $4}' | sort -u | wc -l | tr -d ' ')
        tps_mean=$(awk -v m="$mean" 'BEGIN{printf "%.2f", 1e3/m}')
        tps_p50=$(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}')
        echo "ROW r$r ours d=$dep n=$N ctx=$ctx | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok | nodes ${nodes:-?} | wall $((t1 - t0))s"
        sums+=("ours|$dep|$r|$tps_mean|$tps_p50")
        ;;
    esac
  done
done
echo
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, the references"
echo "    llama-bench's own mean over the N steps — the cross-engine ratio reads these. The p50"
echo "    column is ours only. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  k = $1 " d=" $2; s[k] += $4; n[k]++; if ($5 != "") { sp[k] += $5; np[k]++ }
  if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
} END { for (k in s) {
  spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
  printf "mean %-12s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)\n", k, s[k] / n[k], mn[k], mx[k], spread, (np[k] ? sprintf("%.2f tok/s(p50)", sp[k] / np[k]) : ""), n[k] } }' | sort
echo
echo "=== ours / reference per depth: each round's ratio of the pair measured in that round (arms"
echo "    that ran more than once in a round are averaged first), their mean with its 95 % interval"
echo "    (Student t, rounds - 1 degrees of freedom; 2.0 past 21 rounds), and the ratio of the arm means ==="
deps=$(printf '%s\n' "${ARMS[@]}" | sed 's/^[a-z]*://' | sort -un | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | awk -F'|' -v deps="$deps" -v rounds="$ROUNDS" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
} END {
  split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086", t, " ")
  nd = split(deps, d, " ")
  split("ik ikdef lcpp", refs, " ")
  for (i = 1; i <= nd; i++) for (j = 1; j <= 3; j++) {
    ref = refs[j]
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
    printf "ratio d=%-5s ours/%-5s  mean %.4f %s (n=%d)  of means %.4f  per round:%s\n", d[i], ref, m, ci, c, (as["ours" SUBSEP d[i]] / an["ours" SUBSEP d[i]]) / (as[ref SUBSEP d[i]] / an[ref SUBSEP d[i]]), list
  }
}'
witness post
ref_witness
