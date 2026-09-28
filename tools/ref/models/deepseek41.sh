#!/usr/bin/env bash
# shellcheck shell=bash
# The deepseek41 (DeepSeek-V4.1-Flash) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=deepseek41; never executed, and it exports nothing. Same shape as deepseek2.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says deepseek41), so
# `grep deepseek41` finds the port's arch, its metadata keys (deepseek41.engram.*), this profile
# and the oracle set at once.
#
#   V41_MODEL       the V4.1 file this tree runs, shard 1 of the split set: BLOOMERY_V41_MODEL, else the
#                   default below, V41_PUBLIC. The one shell owner of the path (gguf::v41 is the Rust one, with the
#                   same default): tools/box.sh sources this file in every box command, whatever profile
#                   the command picked, and exports V41_MODEL and V41_DIR as BLOOMERY_V41_MODEL and
#                   BLOOMERY_V41_DIR, so a script or a test under another profile opens the same file.
#                   V41_PUBLIC, the public Q3_K_M set, is the default
#   V41_DIR         the directory of V41_MODEL, where every shard lies
#   V41_SET_SUFFIX  what the oracle set names below carry for the file V41_MODEL names: _plain for
#                   V41_PUBLIC, nothing for any other file. A set
#                   opened for another file than it was dumped from is caught by its `# model` line
#   MODEL           shard 1 of the split set, V41_MODEL; the loader follows split.count from there.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in deepseek2.sh
#   DSPARK_MODEL    the DSpark draft: the copy whose target_layers names the layers whose attention
#                   input the reference captures. BLOOMERY_DSPARK_MODEL moves it. gate-dspark-read
#                   exports it to the gate under that name; markov-accept passes it as the draft argument
#   IK              moves the whole ik tree. The default is our V4.1 port (PR #2455) with the index-key
#                   fix (PR #2507) on the local branch v41/idxkey-fix, a worktree of the V2-Lite
#                   profile's tree; its V4.1 path has only run on the CPU (-ngl 0). The two profiles'
#                   trees differ, so dump.sh refuses a dump_ref built against the other one ([foreign-lib])
#   REF_CTX         the one context the reference files are produced at
#   REF_SET_CPU     ref_deepseek41 (+ V41_SET_SUFFIX) under $BLOOMERY_DATA — flat, a sibling of ref,
#                   ref_cuda and ref_cuda_v2, because every reader resolves a set as $BLOOMERY_DATA/<one
#                   name> (dump.sh's staging and .old siblings, the gates' BLOOMERY_REF_SET)
#   REF_SET_CUDA    ref_cuda_deepseek41 (+ V41_SET_SUFFIX): a name only — no CUDA set is made for this model
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: the dump pages in hundreds of GiB of weights, so dump.sh runs it under the
#                   machine-wide CPU lease and witnesses it. Not overridable from here.
#   REF_DUMP_ARGS   dumper arguments every dump of this model adds: --defer-experts, which skips the
#                   loader's MAP_POPULATE of the whole file set (more than the page cache holds) and
#                   changes nothing the graph computes; the dump faults in what it touches
#   ref_step_variant  the decode-step variants, below
#
#   IK_NCMOE        how many layers keep their routed experts on the host in IK_GPU_FLAGS
#                   (--n-cpu-moe; which layers is the binary's, IK_GPU_FLAGS below), sized exactly as
#                   LCPP_NCMOE is, per timing card: 33 on the A6000 and
#                   37 on the 3090 when MODEL is V41_PUBLIC [derived, LCPP_NCMOE's arithmetic below; no ik
#                   load has been run at 33 or 37], and unset for any other file, which no count was
#                   sized for — IK_GPU_FLAGS is then unset too, and a runner that reads it stops at the
#                   name (set -u), as for IK_BEST_FLAGS below.
#                   The arithmetic carries over because ik places the same tensors: token_embd and
#                   the engram tables (engram_embd) are made in the host input context
#                   (src/llama-load-tensors.cpp:3326, 3468 at db517b69), engram_wkv/k/q and every other
#                   tensor in the split context, so the card dense bytes and the per-layer expert bytes
#                   are mainline's for each file. ik needs no -nopo twin: its CUDA backend offloads a
#                   MUL_MAT_ID only when ubatch × used experts >= GGML_CUDA_MIN_BATCH_OFFLOAD (32, the
#                   build's CMakeCache) × 384 experts (ggml/src/ggml-cuda.cu:5215-5227), and llama-bench's
#                   default ubatch 512 gives 3,072 < 12,288, so no host expert tensor is copied into the
#                   compute buffer and the 2,713 MiB left at 33 on V41_PUBLIC are what mainline has
#                   [derived]. BLOOMERY_IK_NCMOE (ik-draft.sh) replaces it for one run.
#   IK_GPU_FLAGS    ik's decode on the timing card, placed to mirror design §5 (a) (depth-ds41.sh reads
#                   it): every layer offloaded, and the routed experts of the first IK_NCMOE layers on the
#                   CPU (llama-bench's reading; llama-cli's is below), the rest of the layers' whole on
#                   the card. Plan (a) keeps a prefix of every
#                   layer's experts on the A6000, and ik moves a layer's 384 experts as one tensor, so
#                   the closest it has is whole layers. V41_PUBLIC: plan (a)'s 2,668 card experts are
#                   6.95 layers' worth, 33 keeps 7 (LCPP_NCMOE below).
#                   -t 32 is llama-bench's own default, spelled out;
#                   --defer-experts skips the loader's MAP_POPULATE of a file set larger than the page
#                   cache. No flag sweep has been run: these are "at these flags".
#                   llama-bench's --n-cpu-moe N is a list of CPU buffer-type overrides for layers
#                   0..N-1 (the same thing -ot "...=CPU" gives), not the loader's own ncmoe.
#                   llama-cli (ik-draft.sh) hands N to the loader's ncmoe instead, which counts from the
#                   other end: with one visible device (or N at least the layer count) the last N layers
#                   that carry experts go to the host; with two devices under the default layer split, N
#                   is shared out over the devices in proportion to their layers and each device's share
#                   is again its last layers (ik src/llama-load-tensors.cpp:279-358 at 41b17995, the
#                   Mac checkout; the box's ik trees are other commits). The same IK_GPU_FLAGS thus keep
#                   different layers on the card in llama-bench and in llama-cli, and the prefix
#                   placement above is llama-bench's only.
#   IK_GPU_ENV      the environment of every ik launch with IK_GPU_FLAGS, as NAME=VALUE words for
#                   `env`. GGML_CUDA_NO_PINNED_WEIGHTS=1 is load-bearing: any override to the CPU
#                   makes ik's loader drop mmap for the whole host context
#                   (src/llama-load-tensors.cpp, `use_mmap_buffer &= !has_buft_overrides`), and that
#                   context — the IK_NCMOE layers' experts and the engram tables, hundreds of GiB — would
#                   then be one pinned allocation, larger than RAM. With the variable the host
#                   context stays on the file mapping, unpinned, and the staging buffers stay
#                   pinned. GGML_CUDA_NO_PINNED=1 also keeps the mapping, but unpins the staging
#                   buffers too. Never add GGML_CUDA_REGISTER_HOST: it would cudaHostRegister the
#                   whole mapping.
#
#   LCPP            the mainline llama.cpp tree the `lcpp:<D>` arms of depth-ds41.sh and the `lcpp` arm
#                   of ik-draft.sh run: the V4.1 port's branch (ggml-org/llama.cpp PR #28696,
#                   runtime/deepseek41). LCPPBIN moves its llama-bench alone, as IKBIN does ik's
#   LCPP_NCMOE      how many leading layers keep their routed experts on the host (--n-cpu-moe),
#                   sized per timing card from BLOOMERY_TIMING_GPU (the 3090 when it names GPU_3090,
#                   tools/ref/cards.sh; else the A6000, timing-card.sh's default): 33 on the A6000 and 37
#                   on the 3090 when MODEL is V41_PUBLIC, unset for any other file (and with it
#                   LCPP_GPU_FLAGS, LCPP_CLI_FLAGS and LCPP_NCMOE_SWEEP) [all derived, from the file's
#                   header and the card's memory.total — 49,140 MiB and 24,576 MiB, the CardSpecs in
#                   crates/model/src/placement/workstation.rs; no load has been run at 37].
#                   Layer experts, layers 0-1 7,007,109,120 B (down Q5_K), layers 2-39
#                   6,440,878,080 B (down Q4_K). Card dense is every other tensor but token_embd
#                   (the input embedding stays on the host) and the engram tables (the port marks
#                   them TENSOR_READ_LAZY, a host buffer).
#                   V41_PUBLIC: card dense 3,595,917,760 B.
#                     Plan (a)'s card experts on this file: card_experts=2668 (44,750,684,160 B),
#                     6.95 layers' worth. 33 keeps layers 33-39 on the card, 7 × 6,440,878,080 =
#                     45,086,146,560 B (+0.75 % of plan (a)'s bytes; host share of routed work
#                     33/40 = 0.825 against plan (a)'s 1 − 2668/15360 = 0.826).
#                     Card total at 33: 48,682,064,320 B = 46,427 MiB of 49,140 MiB, 2,713 MiB left
#                     for the CUDA context, the KV cache (at most 40 layers × ctx × 512 × f16 — the
#                     port passes one tensor as K and V — 170 MiB at ctx 4,352) and the compute
#                     buffers. At 32 the experts alone are 8 × 6,440,878,080 = 51,527,024,640 B,
#                     the card's whole 49,140 MiB: it does not load.
#                   The 3090 (24,576 MiB, capped at 250 W — power does not move the fit), the same
#                   arithmetic: the most layers whose experts fit beside the card dense with the
#                   A6000's margin at 33 (2,713 MiB, a margin a load has passed). 37 keeps layers
#                   37-39, 3 × 6,440,878,080 = 19,322,634,240 B; card total 22,918,552,000 B = 21,857 MiB,
#                   2,719 MiB left (against usable memory, total less the driver's 400 MiB on the 3090
#                   and 548 MiB on the A6000: 2,319 MiB here, 2,165 at the A6000's 33). At 36 the experts
#                   alone are 4 × 6,440,878,080 = 25,763,512,320 B = 24,570 MiB and the total 27,999 MiB:
#                   it does not load. 37 holds 1,152 routed experts on the card, the gate placement
#                   generate_ds41 runs there (--place gate) 888 (14,894,530,560 B of experts); host share
#                   of routed work 37/40 = 0.925 against 1 − 888/15360 = 0.942.
#   LCPP_GPU_FLAGS  llama-bench's flags for the `lcpp:<D>` arms (tools/llama-bench/llama-bench.cpp's
#                   spellings):
#                   -ngl 999          every layer and the output head on the card
#                   --n-cpu-moe N     LCPP_NCMOE; llama-bench turns it into CPU buffer overrides of
#                                     blk.<i>.ffn_(up|down|gate|gate_up)_(ch|)exps, i < N
#                                     (common/common.h LLM_FFN_EXPS_REGEX), the ik arm's form
#                   -fa on            flash attention (the CUDA kernel has the 512/512 head); auto
#                                     is the default
#                   -t 32             the host's cores (llama-bench's default is
#                                     common_cpu_get_num_math()), spelled out as in IK_GPU_FLAGS
#                   -nopo 1           no op offload. Off by default, a host-resident weight of an
#                                     op with a batch >= 32 is copied to the card per ubatch
#                                     (ggml-cuda.cu device_offload_op), so the compute buffer
#                                     reserved at ubatch 512 must hold a layer's expert tensors
#                                     (up to 3,114,270,720 B each) — more than the 2,713 MiB
#                                     left at 33 on V41_PUBLIC [derived]. A decode step is batch 1 and never
#                                     offloads: only the untimed -d prefill runs on the host
#                                     instead.
#                   No IK_GPU_ENV twin: mainline keeps the host context on the file mapping with
#                   buffer overrides (src/llama-model.cpp: use_mmap_buffer is always true).
#                   No --defer-experts twin either: the loader WILLNEEDs every non-lazy range
#                   (src/llama-mmap.cpp:501-502, the complement of the lazy ranges over each
#                   shard), 262,788,958,788 B of V41_PUBLIC [derived: the nine shards'
#                   347,271,472,288 B, headers included, less the lazy engram_embd tables'
#                   84,482,513,500 B from the headers], on a 270 GB machine whose file cache
#                   peaks near 261.5 GB (the witness's Cached less Shmem) — the witness's page cache and
#                   pgmajfault lines show what it costs. The lazy engram_embd rows it reads one
#                   4 KB fault at a time (MADV_RANDOM on the lazy range, GET_ROWS as one task);
#                   tools/ref/depth-ds41.sh's Order and Discard say how a window keeps them warm.
#   LCPP_CLI_FLAGS  the same placement in common/arg.cpp's spellings, for llama-completion (the
#                   `lcpp` arm of ik-draft.sh): --no-op-offload for -nopo 1, and -fit off —
#                   common's default fit pass would otherwise adjust the arguments not given
#   LCPP_NCMOE_SWEEP  the --n-cpu-moe values the sweep arms `lcpp<K>:<D>` take (depth-ds41.sh):
#                   LCPP_NCMOE and the two above it, on either card; below it the file does not load
#                   (above). An arm whose value does not load is a FAIL row there, and the runner goes on
#   TWO_CARD_PLACEMENT, LCPP_TS  the two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090, tools/ref/timing-card.sh;
#                   V41_PUBLIC only, empty otherwise, and depth-ds41.sh then refuses the mode by name): the
#                   "A6000+3090" table's mainline line, LCPP_NCMOE 30 and -ts LCPP_TS (36.5/4.5) in
#                   LCPP_GPU_FLAGS; TWO_CARD_PLACEMENT says what they place. llama-bench's -sm layer (its
#                   default) gives device d the layers il whose il / 41 is below the d-th cumulative -ts
#                   fraction — 40 layers and the output as a 41st slot, upper_bound on the normalized split
#                   (src/llama-model.cpp:1521-1546 at 53ed051ce; the V4.1 branch 5210c7c5e has the same code)
#                   — the A6000 device 0 (first in CUDA_VISIBLE_DEVICES) and the 3090 device 1. --n-cpu-moe K
#                   keeps layers 0..K-1's experts on the host, so the card-expert layers are the last 40 - K
#                   and the 3090 takes the last of them and the output. The 3090 holds at most 3 expert
#                   layers (4 × 6,440,878,080 B = 24,570 MiB, more than its usable 24,176), the A6000 at most
#                   7 (8 of them are the whole card): K = 30, the A6000 layers 0-36 (30-36 with experts),
#                   the 3090 layers 37-39 and the output. 36.5/4.5 puts the boundary at 0.890, between slot
#                   36 (0.878) and slot 37 (0.902); 37/4 would put it on slot 37's own value, an equal float
#                   under upper_bound. Each card holds at most its one-card load's bytes: the A6000 7 expert
#                   layers and at most all 3,429 MiB of card dense (46,427 MiB at 33, a load that ran, 2,713
#                   left), the 3090 3 and at most all the dense and the head (21,857 MiB, 2,319 left of its
#                   usable memory: the 37 arithmetic above, never loaded); each card now has a compute
#                   buffer of its own, for its layers [derived; no two-card load has been run]. Host share
#                   of routed work 30/40 = 0.75. The fit arms drop both and let llama-bench's fit place over
#                   the two cards (common/fit.cpp sets tensor_split per device). ik has no two-card line:
#                   its -ts is a byte split over its own layer sizes (get_layer_sizes, src/llama.cpp:4575-4640
#                   at db517b69), not a slot split, and whether those count a host-overridden expert is
#                   unread; the public reference is llama.cpp. LCPP_CLI_FLAGS carries the same -ts in the mode
#                   (ik-draft.sh's lcpp arm, llama-completion, whose -ts reads `/` as llama-bench's does)
#
# Deliberately unset: IK_BEST_FLAGS and REF_PROMPTS. No CPU flag sweep and no prompt set exist for
# this model, and every script that reads them runs under `set -u`, so such a script stops at the
# unset name instead of running ik at flags nobody measured.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=deepseek41
: "${IK:=/home/user/ik-idxkey}"
V41_PUBLIC=/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf
V41_MODEL=${BLOOMERY_V41_MODEL:-$V41_PUBLIC}
V41_DIR=${V41_MODEL%/*}
V41_SET_SUFFIX=
[ "$V41_MODEL" != "$V41_PUBLIC" ] || V41_SET_SUFFIX=_plain
MODEL=${BLOOMERY_REF_MODEL:-$V41_MODEL}
DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf}
: "${IK_GPU_ENV=GGML_CUDA_NO_PINNED_WEIGHTS=1}"
: "${LCPP:=/home/user/llama.cpp-v41}"
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
# The timing card decides the references' --n-cpu-moe (LCPP_NCMOE above): the 3090's UUID from cards.sh.
__v41_dir=${BASH_SOURCE[0]%/*}
[ "$__v41_dir" != "${BASH_SOURCE[0]}" ] || __v41_dir=.
# shellcheck source=tools/ref/cards.sh
source "$__v41_dir/../cards.sh"
unset __v41_dir
TWO_CARD_PLACEMENT=
if [ "$MODEL" = "$V41_PUBLIC" ]; then
  if [ "${BLOOMERY_TIMING_CARDS:-}" = a6000+3090 ]; then
    # ik has no two-card arm; its one-card count keeps IK_GPU_FLAGS defined for the runner's lines.
    : "${IK_NCMOE:=33}"
    : "${LCPP_NCMOE:=30}"
    : "${LCPP_TS:=36.5/4.5}"
    TWO_CARD_PLACEMENT="lcpp --n-cpu-moe $LCPP_NCMOE -ts $LCPP_TS: the A6000 (device 0) layers 0-36, 30-36 with their experts; the 3090 (device 1) layers 37-39 with their experts and the output [derived, models/deepseek41.sh]"
  elif [ -n "${BLOOMERY_TIMING_GPU:-}" ] && [ "$BLOOMERY_TIMING_GPU" = "${GPU_3090:-}" ]; then
    : "${IK_NCMOE:=37}"
    : "${LCPP_NCMOE:=37}"
  else
    : "${IK_NCMOE:=33}"
    : "${LCPP_NCMOE:=33}"
  fi
fi
if [ -n "${IK_NCMOE:-}" ]; then
  : "${IK_GPU_FLAGS:=-ngl 999 --n-cpu-moe $IK_NCMOE -t 32 --defer-experts}"
fi
if [ -n "${LCPP_NCMOE:-}" ]; then
  : "${LCPP_GPU_FLAGS:=-ngl 999 --n-cpu-moe $LCPP_NCMOE -fa on -t 32 -nopo 1${LCPP_TS:+ -ts $LCPP_TS}}"
  : "${LCPP_CLI_FLAGS:=-ngl 999 --n-cpu-moe $LCPP_NCMOE -fa on -t 32 --no-op-offload -fit off${LCPP_TS:+ -ts $LCPP_TS}}"
  : "${LCPP_NCMOE_SWEEP:=$LCPP_NCMOE $((LCPP_NCMOE + 1)) $((LCPP_NCMOE + 2))}"
fi
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_deepseek41$V41_SET_SUFFIX}"
: "${REF_SET_CUDA:=ref_cuda_deepseek41$V41_SET_SUFFIX}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints for
# shard 1 (it loads the vocabulary only). Five ids and no BOS: the file sets
# tokenizer.ggml.add_bos_token to false, so nothing is prepended (its BOS would be id 0).
# Changing them invalidates the whole set.
REF_TOKENS=671,6102,294,8760,344
REF_DUMP_LEASE=1
REF_DUMP_ARGS=(--defer-experts)

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-v41 <variant>`): one decode step dumped
# after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh refuses
# REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets
#   STEP_SET           the set's name under $BLOOMERY_DATA
#   STEP_CTX           -c
#   STEP_PREFILL       tokens evaluated quietly first; the step is the next one, at position STEP_PREFILL
#   STEP_TOKENS        the ids as a comma list, or
#   STEP_TOKENS_FILE   a file of ids, one per line, whose first STEP_PREFILL + 1 are the sequence, and
#   STEP_TOKENS_SHA256 that file's sha256: dump.sh refuses a file that no longer has it
#   STEP_ARGS          further dumper arguments
# and returns 1 for a name it does not know. Two suffixes, in either order, change one thing each
# and add it to the set name: `-unfused` turns ik's fused indexer top-k op off (--no-fused-idx-topk),
# so the indexer's scores and its TOP_K are nodes; `-every-node` runs the prefill under the dumped
# schedule instead of the fused one (--prefill-every-node), so the caches the step reads carry a
# dumped prefill's arithmetic — step4-every-node is the variant a batch set's last token can check.
# Every set name then ends in V41_SET_SUFFIX.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1     the indexer top-k overridden to 64, so the indexer, the row gather and mask_to_idx run at a
#          short prefix: prefill 301, the step at 301 (a csa group completes there), -c 512
#   d1n    d1 without the override: the file's top-k (512) at prefill 301, so neither kind builds an indexer
#          (top_k >= pad256(n_vis) for csa and hca) while the window mask already hides keys older than 128 —
#          the step set of a chain that has no indexer yet
#   d2     the model's own top-k: prefill 1,025, the step at 1,025, -c 2048
# d1, d1n and d2 read the prose stream the router trace ran over (router/prose/MANIFEST.tsv names it):
# /root/bloomery-data/engram/corpus-prose-all.ids, 4,670,384 lines, sha256
# f7785d0fc84a4a7e3673a220120be8c6735ab226084b2a506d16412ec64f71a0 — d1 and d1n its first 302 ids, d2 its
# first 1,026.
ref_step_variant() {
  local name=$1 unfused=0 every_node=0
  while :; do
    case $name in
      *-unfused)    unfused=1;    name=${name%-unfused} ;;
      *-every-node) every_node=1; name=${name%-every-node} ;;
      *) break ;;
    esac
  done
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_deepseek41_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1)    STEP_SET=ref_deepseek41_d1;    STEP_CTX=512;  STEP_PREFILL=301
           STEP_ARGS=(--override-kv deepseek41.attention.indexer.top_k=int:64) ;;
    d1n)   STEP_SET=ref_deepseek41_d1n;   STEP_CTX=512;  STEP_PREFILL=301 ;;
    d2)    STEP_SET=ref_deepseek41_d2;    STEP_CTX=2048; STEP_PREFILL=1025 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1|d1n|d2) STEP_TOKENS_FILE=$BLOOMERY_DATA/engram/corpus-prose-all.ids
               STEP_TOKENS_SHA256=f7785d0fc84a4a7e3673a220120be8c6735ab226084b2a506d16412ec64f71a0 ;;
  esac
  if [ "$unfused" = 1 ]; then
    STEP_SET=${STEP_SET}_unfused
    STEP_ARGS+=(--no-fused-idx-topk)
  fi
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
  STEP_SET=${STEP_SET}${V41_SET_SUFFIX}
}
