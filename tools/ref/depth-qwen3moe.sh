#!/usr/bin/env bash
# Qwen3-30B-A3B decode by depth and prefill by prompt length with the whole model on the timing
# card, four engines in one lease (run on the box, lead-only): our engine (generate_qwen3moe
# --time), ik (llama-bench -gp D,N; -p P -n 0 for prefill), mainline llama.cpp (llama-bench -d D;
# -p P -n 0) and mistral.rs (mistralrs bench --depth D; --prompt-len P for prefill), alternated arm
# by arm.
#
#   BLOOMERY_MODEL=qwen3moe tools/box.sh 'bash tools/ref/depth-qwen3moe.sh 6 ik:6 lcpp:6'
#   just depth-gpu-qwen3moe 6 ik:6 ikdef:6 lcpp:6 1024 ik:1024 ikdef:1024 lcpp:1024 4096 ik:4096 ikdef:4096 lcpp:4096
#   just depth-gpu-qwen3moe 6 lcpp:6 mrs:6 4096 lcpp:4096 mrs:4096
#   just depth-gpu-qwen3moe 512 ikpp:512 lcpppp:512 mrspp:512 4096 ikpp:4096 lcpppp:4096 mrspp:4096
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-qwen3moe 6 lcpp:6 mrs:6    # the command lines, no lease, no load
#   just depth-gpu-qwen3moe 6 bin:/root/repo/bloomery-<track>-base/target/release/generate_qwen3moe:6 lcpp:6
#
# depth-ds41.sh's shape, and it blocks the same failure: a ratio read at one depth and quoted as
# "decode is faster" — a step's attention term grows with the cached keys, so the depth goes into
# every number (`tok/s @ n=N, depth D, <card>`) and the ratio table at the end is per depth.
#
# Arms, in lease order (the order rotates by one slot each round — the position bias ab-decode.sh
# names):
#   <D>       ours, D >= 1: generate_qwen3moe --tokens <lcg_prompt D> -n N --ctx C --time. The D
#             fed ids are prefilled untimed (Qwen3moeModel::prefill: D >= 9 in ubatches of up to 512
#             through the grouped GEMM, a tail of at most 8 and a D <= 8 as one pass; the `step 0`
#             line's `plan=` names the units), from lease.sh's lcg_prompt: 100000, then ids in
#             [1000, 91000), all inside this vocabulary; generated token 0 comes out of the last
#             pass and the N - 1 steps after it are timed. C is D + N rounded up to 256: both
#             llama-benches size n_ctx to D + N for this test and pad it to 256 under flash
#             attention (ik: GGML_PAD(n_ctx, llama_kv_cache::get_padding(flash_attn)) in
#             src/llama.cpp; mainline: GGML_PAD(n_ctx, 256) in src/llama-context.cpp), so the
#             caches have one height (mistral.rs: below).
#             Our flash grid is fixed by C (flash_gqa::segments_for), so C goes into the row.
#             BLOOMERY_GEN_CTX fixes C for every ours arm instead, e.g. at a serving height.
#             The row also carries `pp_tok/s <v> (n=D, passes=K)` from the binary's `time prompt`
#             row, the wall of that prefill (Prefill below): one arm reports both the prefill of
#             D ids and the decode at depth D.
#   bin:<path>:<D>  a second generate_qwen3moe (an absolute path on the box, a base tree's build) at
#             depth D with the ours arm's command line, row label `bin:<basename of its tree>` (the
#             tree is the path above `target/`). Beside a plain `<D>` arm it is the same-lease A/B of
#             two builds. It is a base by construction, so its freshness is not asked; its tree line
#             (sha256, HEAD, dirty files) is printed with the references'. Every label is its own
#             engine in the per-arm means and in the ratio table (ours / each other label).
#   ik:<D>    ik: llama-bench -p 0 -n 0 -gp D,N -r 1 $IK_GPU_FLAGS (D = 0: plain tg N). The D-token
#             prefill runs first and untimed: ik restarts its clock after it.
#   ikdef:<D> the same at $IK_GPU_DEFAULT_FLAGS: llama-bench's own defaults with every layer on the
#             card. IK_GPU_FLAGS opts into two merges nobody has timed on this model; this arm,
#             interleaved with ik:<D>, says whether they are ik's faster set.
#   lcpp:<D>  mainline: llama-bench -p 0 -n N -d D -r 1 $LCPP_GPU_FLAGS (D = 0: plain tg N).
#             Mainline has no -gp; -d prefills D tokens before its clock starts. Its label is
#             `tgN @ dD`, ik's `tgN@ppD`.
#   mrs:<D>   mistral.rs, D >= 1: mistralrs bench -f <MODEL> --prompt-len 0 --gen-len N --depth D
#             --iterations 1 --warmup 1 --max-seq-len C --pa-context-len C+32 $MRS_FLAGS. One request
#             of D prompt ids with greedy sampling and EOS stop off; its clock (mistralrs-cli
#             src/commands/bench.rs, recv_measurement) runs from the first streamed token to the
#             last, over N - 1 intervals, so the timed tokens sit at the positions ours times. The
#             warmup is one whole discarded request, as llama-bench's own warmup run is.
#             --pa-context-len sizes the PagedAttention pool (on by default on CUDA) to the
#             other engines' cache height C instead of 90 % of the free memory: the pool keeps
#             one 32-token block back, so C + 32 leaves C usable. --max-seq-len C is the
#             automatic device mapper's length, which only places layers. The weights stay in
#             the file's quantized types; activations are bf16 (`DType selected`) and so is the
#             pool (`KV cache type`) — load lines echoed with the row. mistral.rs
#             refuses --depth 0 with a decode length (bench.rs run_bench), so D = 0 is refused here.
#             Its row reads the `Decode (N tokens @ dD)` row of the bench's table (T/s, one decimal).
#   mrspa0:<D> the same with --paged-attn off in place of --pa-context-len: mistral.rs's own KV
#             cache and attention path, interleaved with mrs:<D>, says which is its faster set.
#             That cache is sized by mistral.rs, not by C: this arm is outside the one-height
#             statement above.
#   ikpp:<P>  ik's prefill: llama-bench -p P -n 0 -r 1 $IK_GPU_FLAGS, the ik:<D> arm's binary and
#             flags; the row reads llama-bench's `ppP` row as `tok/s(pp) … @ n=0, prompt P`.
#   lcpppp:<P> mainline's prefill: llama-bench -p P -n 0 -r 1 $LCPP_GPU_FLAGS, the lcpp:<D> arm's.
#   ikpp<U>:<P>, lcpppp<U>:<P>  the same with -ub U -b max(U, 2048): the ubatch lever, not a default
#             (Prefill below). Refused when the profile's flags already name -ub or -b.
#   mrspp:<P> mistral.rs's prefill: mistralrs bench -f <MODEL> --prompt-len P --gen-len 1
#             --iterations 1 --warmup 1 --max-seq-len C --pa-context-len C+32 $MRS_FLAGS, C = P + 1
#             rounded up to 256 (or BLOOMERY_GEN_CTX): a gen length below 2 skips the decode case,
#             and the one request of P ids plus its first token must fit --max-seq-len
#             (bench.rs run_bench). Its row reads the `TTFT (P input tokens)` row of the bench's
#             table: T/s = P / TTFT.
# ik and mainline feed std::rand() ids, prefill and steps alike; mistral.rs feeds prompt ids 1000 +
# (start + i) mod 2048 and then its greedy continuation; ours feeds its greedy continuation, which
# is why our row carries distinct_tokens. Every arm is one process — a model load, the prefill, N
# steps (mistral.rs: its warmup request, then the timed one) — and the four engines open the one
# file.
#
# Flags. The ours, ik and lcpp arms hold every layer, the output head and an f16 K/V cache on the
# card (llama-bench's -ctk/-ctv default is f16; ours keeps f16 planes), so a step reads the same
# weight and cache bytes in each. The input embedding differs: both references keep it on the host and
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
#   mrs   --format gguf  the file's format, spelled out (the suffix would pick it)
#         left out: a subcommand (bench's own options take the model; `auto` is the default path);
#         -n/--device-layers and --topology (one visible card, automatic mapping puts every layer
#         on it); --isq, --quant, --dtype (another computation, or the file's own types); --cpu;
#         --pa-block-size, --pa-cache-type (their defaults: 32 tokens, the compute dtype); --seed
#         (greedy sampling); --max-batch-size (one sequence). The binary does not print its cargo
#         features; the witness records its sha256, tree and version, and the load lines where
#         its layers and cache went.
#
# The trees are the profile's IK, LCPP and MRS. The witness prints each binary's sha256 with its
# tree's HEAD and dirty count, and each reference row the binary's own `build:` line (mistral.rs:
# its --version line) — a tree can move after its binary was built — and the device llama-bench
# opened.
#
# Paging. The witness prints the page cache and the major fault count before and after every arm:
# the file is 18.6 GB, and another round's host set can evict it between two arms.
#
# Prefill. pp_tok/s is P over the wall of processing a P-token prompt, in each engine's terms:
#   ours      generate_qwen3moe's `time prompt` row: the wall of Qwen3moeModel::prefill through the
#             readback of its token — P >= 9 in ubatches of up to 512 through the grouped GEMM
#             (`kind=gemm`), a P <= 8 as one pass (`kind=prefill`), `passes` the units. The prefill
#             buffers are allocated at load, so the wall carries no allocation. The arm's cache
#             height is C (D + N rounded up to 256), not P.
#   ik, lcpp  llama-bench's pp test: llama_decode over the P ids in batches of -b and ubatches of
#             -ub, then one synchronize (test_prompt), one repetition, llama-bench's own value, at
#             n_ctx = P padded to 256. Both trees default to -ub 512 -b 2048; the row names them.
#   mrs       the TTFT case: from sending the request to its first streamed token
#             (run_single_bench, recv_measurement), so it also holds the request's scheduling and
#             the first sample, as ours holds the readback of token 0. mistral.rs batches by its
#             scheduler's defaults, max_num_batched_tokens 4096 per step and
#             max_prefill_chunk_tokens 512 while a decode is resident (mistralrs-cli
#             src/args/mod.rs); bench sets neither, so mrspp has no ubatch lever.
# The arms do not start alike. The warmup before the timed run is a 1-token prompt in ik
# (examples/llama-bench/llama-bench.cpp:2230), the whole prompt in mainline
# (tools/llama-bench/llama-bench.cpp:2382) and one whole request in mistral.rs (--warmup 1); ours
# has none. So ik's and our rows carry first-touch costs the other two do not.
# The ubatch lever: every expert sits on the card here, and a ubatch reads each routed expert's
# weights once whatever its row count, so a larger ubatch shares that read among more tokens
# [derived]. ikpp<U> / lcpppp<U> interleaved with the default arm find the references' faster
# setting; the line to beat is a reference at its fastest flags.
#
# Co-tenants. A compute process on the other card is recorded ([other-busy], timing-card.sh), and
# the arm's row ends in ` [other-busy]`; the closing summary counts those rows. This runner has no
# CPU guard, so no row carries [cpu-busy]. A compute process on the timing card as an arm starts
# holds the arm until it is gone, and after 10 minutes stops the runner with rc 75 and no summary
# (guard_timing below).
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 4),
# BLOOMERY_GEN_WARM (generate_qwen3moe --warm), BLOOMERY_GEN_CTX (above; mrs arms take the same C),
# BLOOMERY_GEN_BIN (default target/release/generate_qwen3moe), BLOOMERY_ARM_BOUND (seconds one arm
# may run, default 900: a hung arm fails the runner with rc 124/137 instead of holding the lease),
# BLOOMERY_DRY=1 (print each arm's command line, the binaries' tree lines and the rotation, then
# exit 0 before the lease: nothing is loaded and nothing is timed).
set -uo pipefail
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_DEFAULT_FLAGS, LCPP, LCPPBIN, LCPP_GPU_FLAGS,
# MRS, MRSBIN, MRS_FLAGS); tools/box.sh exports its MODEL to our binary as BLOOMERY_REF_MODEL, so
# the four engines open one file.
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
DRY=${BLOOMERY_DRY:-}
case $GEN_CTX in
  *[!0-9]* | 0) echo "depth-qwen3moe.sh: BLOOMERY_GEN_CTX is a positive integer, got '$GEN_CTX'" >&2; exit 64 ;;
esac
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6 lcpp:6)
ours=0 ik=0 lcpp=0 mrs=0
# Per arm, by its index in ARMS: the kind (ours, ref or bin), the depth, the row label and the binary
# (ours and bin).
A_KIND=() A_DEP=() A_LABEL=() A_BIN=()
arm_usage() {
  echo "depth-qwen3moe.sh: arm '$1' is <D>, ik:<D>, ikdef:<D>, lcpp:<D>, mrs:<D>, mrspa0:<D>, ikpp[<U>]:<P>, lcpppp[<U>]:<P>, mrspp:<P> or bin:<path>:<D>" >&2
  exit 64
}
# A prefill arm's engine (ikpp[<U>], lcpppp[<U>], mrspp), and its ubatch lever U (empty: the default).
pp_eng() { case $1 in ikpp* | lcpppp* | mrspp) return 0 ;; *) return 1 ;; esac; }
pp_ub() { local u=${1#ikpp}; u=${u#lcpppp}; echo "${u#mrspp}"; }
for a in "${ARMS[@]}"; do
  kind=ref eng=${a%%:*} dep=${a#*:} label='' bin=''
  case $a in
    bin:*)
      kind=bin bin=${a#bin:}
      dep=${bin##*:} bin=${bin%:*}
      case $bin in /*) ;; *) arm_usage "$a" ;; esac
      tree=${bin%/target/*}
      [ "$tree" != "$bin" ] || tree=${bin%/*}
      label=bin:${tree##*/}
      ;;
    *:*)
      case $eng in
        ik | ikdef | ikpp | ikpp[1-9]*) ik=1 ;;
        lcpp | lcpppp | lcpppp[1-9]*) lcpp=1 ;;
        mrs | mrspa0 | mrspp) mrs=1 ;;
        *) arm_usage "$a" ;;
      esac
      label=$eng
      ;;
    *) kind=ours dep=$a bin=$BIN label=ours ours=1 ;;
  esac
  case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
  case $kind:$eng:$dep in
    ref:mrs:0 | ref:mrspa0:0) echo "depth-qwen3moe.sh: arm '$a': mistral.rs refuses --depth 0 with a decode length" >&2; exit 64 ;;
  esac
  if [ "$kind" = ref ] && pp_eng "$eng"; then
    ub=$(pp_ub "$eng")
    case $ub in *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-qwen3moe.sh: arm '$a': a prompt of 0 ids has no prefill to time" >&2; exit 64; }
    if [ -n "$ub" ]; then
      case $eng in ikpp*) flags=$IK_GPU_FLAGS ;; *) flags=$LCPP_GPU_FLAGS ;; esac
      case " $flags " in
        *" -ub "* | *" --ubatch-size "* | *" -b "* | *" --batch-size "*)
          echo "depth-qwen3moe.sh: arm '$a': the profile's flags already set the batch sizes ($flags); the ubatch lever would add a second value" >&2
          exit 64
          ;;
      esac
    fi
  fi
  # Our prompt is one argument of D ids of up to six characters each; the kernel caps one
  # argument at 128 KiB.
  if [ "$kind" != ref ] && { [ "$dep" -lt 1 ] || [ "$dep" -gt 20000 ]; }; then
    echo "depth-qwen3moe.sh: our arm '$a' needs 1 <= D <= 20000 (D fed ids in one --tokens argument)" >&2
    exit 64
  fi
  A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_BIN+=("$bin")
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
  echo "depth-qwen3moe.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS (rc $rc; ROUNDS is a positive integer)" >&2
  exit "$rc"
}
# Each engine's binary matters only to its own arms: a reference-only run neither builds ours (the
# recipe skips the build) nor reads it, so its freshness is not asked. A dry run asks nothing of
# our binary: it prints the command line it would run.
if [ "$ours" = 1 ] && [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
# A bin:<path> arm's binary is checked where its tree line is taken, below.
if [ "$ik" = 1 ]; then [ -x "$IKBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $IKBIN" >&2; exit 2; }; fi
if [ "$lcpp" = 1 ]; then [ -x "$LCPPBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $LCPPBIN" >&2; exit 2; }; fi
if [ "$mrs" = 1 ]; then [ -x "$MRSBIN" ] || { echo "depth-qwen3moe.sh: no mistralrs at $MRSBIN" >&2; exit 2; }; fi
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
IK_LINE='' LCPP_LINE='' MRS_LINE='' MRS_VERSION='' BIN_LINES=()
[ "$ik" = 0 ] || IK_LINE=$(tree_line "$IKBIN" "$IK")
# shellcheck disable=SC2153 # LCPP is the profile's, which shellcheck does not follow
[ "$lcpp" = 0 ] || LCPP_LINE=$(tree_line "$LCPPBIN" "$LCPP")
if [ "$mrs" = 1 ]; then
  # shellcheck disable=SC2153 # MRS is the profile's, as LCPP
  MRS_LINE=$(tree_line "$MRSBIN" "$MRS")
  MRS_VERSION=$("$MRSBIN" --version 2>&1 | head -n 1)
  MRS_LINE="$MRS_LINE version=${MRS_VERSION:-?}"
fi
for i in "${!ARMS[@]}"; do
  [ "${A_KIND[$i]}" = bin ] || continue
  b=${A_BIN[$i]}
  [ -x "$b" ] || { echo "depth-qwen3moe.sh: arm '${ARMS[$i]}': no binary at $b" >&2; exit 2; }
  t=${b%/target/*}
  [ "$t" != "$b" ] || t=${b%/*}
  BIN_LINES+=("${A_LABEL[$i]}: $(tree_line "$b" "$t")")
done
ref_witness() {
  [ -z "$IK_LINE" ] || echo "    ik: $IK_LINE"
  [ -z "$LCPP_LINE" ] || echo "    lcpp: $LCPP_LINE"
  [ -z "$MRS_LINE" ] || echo "    mrs: $MRS_LINE"
  [ ${#BIN_LINES[@]} -eq 0 ] || printf '    %s\n' "${BIN_LINES[@]}"
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

# ref_cmd <engine> <depth or prompt length>: the reference arm's binary, arguments and row label,
# into REF_BIN, REF_ARGS and REF_LABEL, and for a prefill arm the batch sizes its row names into
# REF_BATCH. The flags are word-split on purpose: the profile keeps them as one string. A prefill
# arm is its decode twin's binary and flags with the prompt test in place of the decode test.
ref_cmd() {
  local eng=$1 dep=$2 flags ctx ub
  REF_BATCH=
  case $eng in
    ik | ikdef)
      REF_BIN=$IKBIN flags=$IK_GPU_FLAGS
      [ "$eng" = ik ] || flags=$IK_GPU_DEFAULT_FLAGS
      if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n 0 -gp "$dep,$N"); REF_LABEL="tg$N@pp$dep |"; fi
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    lcpp)
      REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS
      if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n "$N" -d "$dep"); REF_LABEL="tg$N @ d$dep |"; fi
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    ikpp* | lcpppp*)
      case $eng in
        ikpp*) REF_BIN=$IKBIN flags=$IK_GPU_FLAGS ;;
        *) REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS ;;
      esac
      ub=$(pp_ub "$eng")
      REF_ARGS=(-p "$dep" -n 0) REF_LABEL="pp$dep |" REF_BATCH="ub 512 b 2048 (llama-bench defaults)"
      if [ -n "$ub" ]; then
        REF_ARGS+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))")
        REF_BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
      fi
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    mrspp)
      REF_BIN=$MRSBIN flags=$MRS_FLAGS
      ctx=${GEN_CTX:-$(((dep + 1 + 255) / 256 * 256))}
      # shellcheck disable=SC2206
      REF_ARGS=(bench $flags -f "$MODEL" --prompt-len "$dep" --gen-len 1 --iterations 1 --warmup 1 --max-seq-len "$ctx" --pa-context-len "$((ctx + 32))")
      REF_LABEL="TTFT ($dep input tokens)"
      REF_BATCH="scheduler defaults: max_num_batched_tokens 4096, max_prefill_chunk_tokens 512 (bench sets neither)"
      ;;
    mrs | mrspa0)
      REF_BIN=$MRSBIN flags=$MRS_FLAGS
      ctx=${GEN_CTX:-$(((dep + N + 255) / 256 * 256))}
      # shellcheck disable=SC2206
      REF_ARGS=(bench $flags -f "$MODEL" --prompt-len 0 --gen-len "$N" --depth "$dep" --iterations 1 --warmup 1 --max-seq-len "$ctx")
      # The pool holds (blocks - 1) whole 32-token blocks of the context asked for (the scheduler keeps
      # one back: paged_attention/mod.rs, available_context_tokens), so one block more than C.
      if [ "$eng" = mrs ]; then REF_ARGS+=(--pa-context-len "$((ctx + 32))"); else REF_ARGS+=(--paged-attn off); fi
      REF_LABEL="Decode ($N tokens @ d$dep)"
      ;;
  esac
}

# ref_val <engine> <label>: the arm's tok/s from its output on stdin. llama-bench prints a markdown
# table (`| … | t/s ± sd |`); mistralrs bench a box-drawn one (`│ Decode (N tokens @ dD) ┆ T/s ± sd
# ┆ … ms TPOT │`) whose first `<number> ± ` is the T/s cell. Its log lines carry ANSI colour codes.
ref_val() {
  case $1 in
    mrs | mrspa0 | mrspp) sed 's/\x1b\[[0-9;]*m//g' | grep -F "$2" | grep -oE '[0-9][0-9.]* ± ' | head -n 1 | sed 's/ ± //' ;;
    *) grep -F "$2" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g' ;;
  esac
}

# One reference arm: its binary on its flags, the row, and the sum.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 raw rc val build dev t0 t1 fail
  ref_cmd "$eng" "$dep"
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  raw=$(timeout --kill-after=10 "$BOUND" "$REF_BIN" "${REF_ARGS[@]}" 2>&1)
  rc=$?
  t1=$(date +%s)
  witness "post r$r $eng d=$dep"
  val=$(echo "$raw" | ref_val "$eng" "$REF_LABEL")
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    # The whole output goes to a file: the loader's reason for a failed load is many lines
    # above the tail.
    fail=${TMPDIR:-/tmp}/depth-qwen3moe-$eng-d$dep-r$r.log
    echo "$raw" > "$fail"
    echo "r$r $eng d=$dep produced no '${REF_LABEL% |}' row (rc $rc); full output: $fail" >&2
    echo "$raw" | tail -n 20 >&2
    exit 1
  fi
  case $eng in
    mrs | mrspa0 | mrspp)
      # The load lines that say what ran (weight dtype, layer placement, the KV cache and its
      # length, the decode graphs, the tree), then the bench's own table.
      echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'DType selected|Layers [0-9]+-[0-9]+: |PagedAttention (KV cache type|with block)|CUDA decode graphs|git revision' \
        | sed 's/^[^ ]* *INFO [^ ]* //' | sed "s/^/    $eng load /"
      echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^│' | sed "s/^/    $eng table /"
      build=$MRS_VERSION
      dev=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/.*Layers \([0-9]*-[0-9]*: .*\)$/layers \1/p' | head -n 1)
      ;;
    *)
      # The reference's own table, header and row: its columns name every setting it ran with that
      # differs from its defaults, so the log shows which flags took.
      echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
      build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
      dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
      ;;
  esac
  if pp_eng "$eng"; then
    echo "ROW r$r $eng p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME | $REF_BATCH | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$OTHER_BUSY_TAG"
    pp_sums+=("$eng|$dep|$r|$val|$OTHER_BUSY_TAG")
  else
    echo "ROW r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$OTHER_BUSY_TAG"
    sums+=("$eng|$dep|$r|$val|")
  fi
  count_row
}

# The closing summary's row counts: every ROW line, and those that carried [other-busy].
count_row() {
  n_rows=$((n_rows + 1))
  [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
}

# parse_pp: the `time prompt n=<P> ms=<ms> tok/s=<v> passes=<K> kind=<k>` row of a generate_qwen3moe
# run on stdin, as `<P> <v> <K> <k>`; fields after `kind=` are allowed and dropped; nothing when the
# run printed none (a base tree's build).
parse_pp() {
  sed -nE 's/^time prompt n=([0-9]+) ms=[0-9.]+ tok\/s=([0-9.]+|inf) passes=([0-9]+) kind=([a-z]+)( .*)?$/\1 \2 \3 \4/p' | head -n 1
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
    [ "$kind" = prefill ] || PP_COL="${PP_COL%)}, kind=$kind)"
  elif [ "$1" = bin ]; then
    PP_COL=" | pp_tok/s ? (the binary prints no time prompt row)"
  else
    echo "$2 produced no time prompt row" >&2
    echo "$3" | tail -n 20 >&2
    exit 1
  fi
}

# One arm of a generate_qwen3moe: ours (our binary) or a second binary (bin:). The row and the sum
# under the arm's label.
# ours_arm <index> <round>
ours_arm() {
  local i=$1 r=$2 dep label bin ctx out rc t0 t1 smoke p50 mean warmcol nodes series h10 t10 uniq_tok tps_mean tps_p50
  dep=${A_DEP[$i]} label=${A_LABEL[$i]} bin=${A_BIN[$i]}
  ctx=${GEN_CTX:-$(((dep + N + 255) / 256 * 256))}
  witness "pre r$r $label d=$dep n=$N ctx=$ctx"
  t0=$(date +%s)
  out=$(timeout --kill-after=10 "$BOUND" "$bin" --tokens "$(lcg_prompt "$dep")" -n "$N" --ctx "$ctx" --time ${WARM:+--warm "$WARM"} 2>&1)
  rc=$?
  t1=$(date +%s)
  witness "post r$r $label d=$dep n=$N ctx=$ctx"
  if [ $rc -ne 0 ]; then
    echo "r$r $label d=$dep ctx=$ctx FAILED rc=$rc" >&2
    echo "$out" | tail -n 20 >&2
    exit 1
  fi
  smoke=$(echo "$out" | grep -E '^SMOKE ')
  [ -n "$smoke" ] || { echo "r$r $label d=$dep produced no SMOKE line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
  # The prompt_ids line is the whole prompt; the load, capture, step-0 and time prompt lines are
  # the arm's configuration and the prefill's time.
  echo "$out" | grep -E '^(load|capture|step 0|time prompt) '
  pp_col "${A_KIND[$i]}" "r$r $label d=$dep" "$out"
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
  echo "ROW r$r $label d=$dep n=$N ctx=$ctx | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok | nodes ${nodes:-?}$PP_COL | wall $((t1 - t0))s$OTHER_BUSY_TAG"
  count_row
  sums+=("$label|$dep|$r|$tps_mean|$tps_p50")
  [ -z "$PP_N" ] || pp_sums+=("$label|$PP_N|$r|$PP_TPS|$OTHER_BUSY_TAG")
}

# ratio_table <prefix> <keys> <labels> <tagged>: records `label|key|round|value|extra` on stdin; for
# every key and every label but ours, each round's ours / label ratio (arms that ran more than once
# in a round averaged first), their mean with its 95 % interval (Student t at rounds - 1 degrees of
# freedom, T975) and the ratio of the arm means. With tagged = 1 the extra field is
# the row's contention tag, and the line ends with each side's count of it.
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v tagged="$4" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
  if (tagged && $5 ~ /other-busy/) bo[a]++
} END {
  nt = split(t975, t, " ")
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
    if (c < 2) ci = "(one round: no interval)"
    else if (c - 1 > nt) ci = sprintf("(no t quantile for df %d)", c - 1)
    else ci = sprintf("± %.4f", t[c - 1] * sqrt(ss / (c - 1)) / sqrt(c))
    ao = "ours" SUBSEP d[i]; ar = ref SUBSEP d[i]
    busy = tagged ? sprintf("  busy: ours [other-busy %d/%d], %s [other-busy %d/%d]", bo[ao], an[ao], ref, bo[ar], an[ar]) : ""
    printf "%s%-5s ours/%-6s  mean %.4f %s (n=%d)  of means %.4f  per round:%s%s\n", prefix, d[i], ref, m, ci, c, (as[ao] / an[ao]) / (as[ar] / an[ar]), list, busy
  }
}'
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  ref_witness | sed 's/^   /[dry]/'
  for i in "${!ARMS[@]}"; do
    a=${ARMS[$i]}
    dep=${A_DEP[$i]}
    if [ "${A_KIND[$i]}" = ref ]; then
      ref_cmd "${a%%:*}" "$dep"
      echo "[dry] $a: timeout --kill-after=10 $BOUND $REF_BIN ${REF_ARGS[*]}   # row label '${REF_LABEL% |}'${REF_BATCH:+, $REF_BATCH}"
    else
      ctx=${GEN_CTX:-$(((dep + N + 255) / 256 * 256))}
      note=''
      [ "${A_LABEL[$i]}" = ours ] || note="   # row label '${A_LABEL[$i]}'"
      echo "[dry] $a: timeout --kill-after=10 $BOUND ${A_BIN[$i]} --tokens <lcg_prompt $dep> -n $N --ctx $ctx --time${WARM:+ --warm $WARM}$note"
    fi
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
echo "[config] ours: $BIN ctx=${GEN_CTX:-D+N rounded up to 256}"
echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS ikdef flags=$IK_GPU_DEFAULT_FLAGS"
echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS"
echo "[config] mrs: $MRSBIN flags=$MRS_FLAGS (mrs: --pa-context-len C, mrspa0: --paged-attn off)"
echo "[config] prefill: ikpp/lcpppp run llama-bench -p P -n 0 -r 1 at the flags above (<U>: -ub U -b max(U, 2048)), mrspp mistralrs bench --prompt-len P --gen-len 1; ours from its time prompt row"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
ref_witness
guard_other
guard_timing

sums=() pp_sums=()
n_rows=0 other_rows=0
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#ARMS[@]} - 1))); do
    j=$(((i + r - 1) % ${#ARMS[@]}))
    a=${ARMS[$j]}
    guard_other
    guard_timing
    case ${A_KIND[$j]} in
      ref) ref_arm "${a%%:*}" "${a#*:}" "$r" ;;
      *) ours_arm "$j" "$r" ;;
    esac
  done
done
echo
echo "other-busy rows: $other_rows of $n_rows (a compute process on the other card as the arm started)"
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, the references"
echo "    their bench's own mean (llama-bench over the N steps, mistralrs bench over N - 1 intervals)"
echo "    — the cross-engine ratio reads these. The p50 column is ours only. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  k = $1 " d=" $2; s[k] += $4; n[k]++; if ($5 != "") { sp[k] += $5; np[k]++ }
  if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
} END { for (k in s) {
  spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
  printf "mean %-14s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)\n", k, s[k] / n[k], mn[k], mx[k], spread, (np[k] ? sprintf("%.2f tok/s(p50)", sp[k] / np[k]) : ""), n[k] } }' | sort
echo
echo "=== ours / reference per depth: each round's ratio of the pair measured in that round (arms"
echo "    that ran more than once in a round are averaged first), their mean with its 95 % interval"
echo "    (Student t, rounds - 1 degrees of freedom; 2.0 past 21 rounds), and the ratio of the arm means ==="
deps=$(printf '%s\n' "${A_DEP[@]}" | sort -un | tr '\n' ' ')
refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -vx ours | sort -u | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$deps" "$refs" 0
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== prefill per prompt length (tok/s(pp) @ n=0, prompt P, $CARD_NAME). Ours: its time prompt"
  echo "    row (prefill through its token's readback); llama-bench's pp value over one repetition;"
  echo "    mistral.rs P / TTFT over one request. The tag counts the rows that met contention. ==="
  printf '%s\n' "${pp_sums[@]}" | awk -F'|' '{
    k = $1 " p=" $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
    if ($5 ~ /other-busy/) o[k]++
  } END { for (k in s) {
    spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
    printf "mean pp %-14s %8.2f tok/s(pp)  [%s..%s, spread %.2f%%]  (n=%d)  [other-busy %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, n[k], o[k], n[k] } }' | sort
  echo
  echo "=== ours / reference prefill per prompt length: the decode table's statistics over the pp"
  echo "    values, then how many of each side's rows carried [other-busy] ==="
  pp_keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  pp_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -vx ours | sort -u | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$pp_keys" "$pp_refs" 1
fi
witness post
ref_witness
