#!/usr/bin/env bash
# shellcheck shell=bash
# The glm5next (GLM-5.3-Flash) profile: everything the reference engine and the harnesses need that is
# a property of the *model*, not of the machine. Sourced by ref-paths.sh when BLOOMERY_MODEL=glm5next;
# never executed, and it exports nothing. Same shape as qwen3moe.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says glm5next), so `grep glm5next`
# finds this profile, the metadata keys (glm5next.*), the oracle sets and their refset family
# (crates/refset/src/arch/glm5next) at once.
#
#   MODEL           shard 1 of unsloth's UD-Q4_K_XL split set (six shards); the loader follows
#                   split.count from there, and the first shard's full path is the identity every
#                   oracle set states. BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored,
#                   as in qwen3moe.sh
#   IK              moves the whole ik tree. The default is the tree the V4.1, qwen3moe and qwen35moe
#                   profiles name: its build carries this architecture (libllama.so holds glm5next and
#                   "GLM5NEXT: hyper_connection.count is required") and the installed dump_ref links
#                   against it, so one dumper serves every profile ([foreign-lib] in dump.sh otherwise)
#   REF_CTX         the one context the batch set is produced at
#   REF_SET_CPU     ref_glm5next under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_glm5next: a name only — no CUDA set is made for this model yet, and the
#                   file does not fit on one card at -ngl 99 (justfile, dump-ref-glm5next-cuda)
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 199.7 GB split set, most of the page cache — more than the machine can
#                   share with a timed run. Not overridable from here.
#   ref_step_variant  the decode-step variants, below
#
# No REF_DUMP_ARGS: without --defer-experts the loader populates the whole split set, which fits in
# the page cache (unlike V4.1's), so a later dump of the same file reads nothing from the device.
#
#   GLM_PROSE       the prose ids under $BLOOMERY_DATA (the d1k variant's and the timing runner's
#   GLM_PROSE_SHA256  prompt, below) and their sha256
#   GLM_PROSE_FROM  the 0-based index of the first prompt id depth-glm5next.sh feeds: 50,000, past
#                   the router set glm5next-prose (router-trace.sh prose --max-tokens 50000, the file's
#                   first 50,000 ids), so the measured prompt is text no router trace saw
#                   (docs/fair-measure.md 4.2) and every published row keeps the same ids;
#                   71,727 - 50,000 = 21,727 ids remain, more than any context the plan allows
#
# The public lines (tools/ref/depth-glm5next.sh, `just depth-gpu-glm5next`). Mainline llama.cpp does
# not build glm5next; two open PR branches do, each in a tree of its own built with mainline's CMake
# flags. Neither is ik, and ik has no arm here: ik is the oracle, not a public row.
#   LCPP27752       ggml-org/llama.cpp PR #27752 (glm5next, KDA through kimi-linear's ops, the DSA
#                   indexer on the hybrid-idx memory); LCPP27752BIN its llama-bench
#   LCPP27754       PR #27754 (GLM-5-Next); LCPP27754BIN its llama-bench. Its body names two settings
#                   for correct output, both in its arms: NVIDIA_TF32_OVERRIDE=0 (LCPP27754_ENV: the
#                   CUDA backend sets TF32 tensor-op math on every fp32 GEMM) and -fa off (its MLA casts
#                   the f32 latent to f16 before the flash kernel)
#   GLM_NCMOE       the leading layers whose routed experts stay on the host (--n-cpu-moe), per timing
#                   card: 36 on the A6000, 42 on the 3090 [derived from the file's header; no load has
#                   been run]. Both branches skip the NextN block (blk.45) unless an MTP context loads
#                   it (TENSOR_SKIP when !load_mtp), and layers 0-2 are dense, so K counts blocks and
#                   the card holds the experts of blocks K..44. Card dense: every tensor but
#                   token_embd, the routed experts and blk.45 — 8,966,188,280 B. Experts a block
#                   4,378,853,376 B; blocks 11 (5,303,697,408), 12 and 44 (4,699,717,632 each) carry
#                   wider down stacks. A6000 at 36: blocks 36-44, 39,730,544,640 B; card total
#                   48,696,732,920 B = 46,441 MiB of 49,140, 2,151 MiB left beside the driver's 548 —
#                   V4.1's 33 loaded with 2,165. At 35 block 35 adds 4,176 MiB: it does not fit.
#                   The 3090 at 42: blocks 42-44, 13,457,424,384 B; total 21,386 MiB of 24,576, 2,790 MiB
#                   left beside the driver's 400. Host share of routed bytes at 36: blocks 3-35, 4.05 GB
#                   a token against ours' 5.15 GB with every routed expert on the host [derived]
#   (fit twins)     depth-glm5next.sh's lcpp2775xfit / lcpp2775xppfit arms drop -ngl and --n-cpu-moe
#                   and leave the placement to llama-bench's fit (tools/ref/lcpp-fit.sh): every
#                   block's dense part on the card, then the routed experts of whole blocks front to
#                   back, then the up and gate of one more, the rest on the host — the TRAILING blocks,
#                   where --n-cpu-moe 36 holds the leading 3-35. Predicted on the A6000 [derived, not
#                   loaded]: room for experts 49,140 MiB less the driver's 548, the 1,024 margin and the
#                   8,551 of card dense = 39,017 MiB less X, the card's KV, recurrent state and compute
#                   buffer at the arm's n_ctx and ubatch (not measured; the hand-set load leaves it at
#                   most 2,151). Blocks 3-10 whole take 33,408 MiB (4,176 each); block 11 whole (5,058)
#                   needs X <= 551; its up and gate (their types not read; 2,784 at q4_K, 3,666 if the
#                   down is the q4_K one) need X <= 1,943..2,825. So `overridden CPU:101 in blk 11-44
#                   (blk 11: 2)`: block 11's ffn_down.* (its routed and its shared down: the fit's
#                   partial-layer pattern is by name) and blocks 12-44's three expert tensors on the
#                   host; 36,192..37,074 MiB of experts on the card against the hand-set's 37,890; host
#                   routed bytes 139,812..140,694 MiB against 138,996 (+0.6..+1.2 %), 4.07..4.10 GB a
#                   token. X <= 551 instead: `CPU:99 in blk 12-44 (blk 12: 3)`. The NextN block stays
#                   out: the fit counts it only under load_mtp (common/fit.cpp:141 in #27754, 140 in #27752)
#   LCPP27752_GPU_FLAGS, LCPP27754_GPU_FLAGS  llama-bench's flags, V4.1's set
#                   (models/deepseek41.sh LCPP_GPU_FLAGS): -ngl 999, --n-cpu-moe GLM_NCMOE, -t 32 (the
#                   host's cores, spelled out), -nopo 1 (no op offload: at a batch >= 32 the backend
#                   would copy a host expert tensor to the card per ubatch, and the compute buffer
#                   that needs, one block expert tensor of 1.36 GB (gate, q4_K) or more, must then
#                   fit in the 2,151 MiB left beside the ubatch activations) [derived],
#                   and -fa on / -fa off (#27754's correctness setting)
#   LCPP27752_CLI_FLAGS, LCPP27754_CLI_FLAGS  the same placement in llama-completion's spellings, for
#                   the build smoke: --no-op-offload, and -fit off (common's fit pass would move the
#                   arguments not given)
#   LCPP27752SRV, LCPP27754SRV  each branch's llama-server: the server arms (`lcpp2775xsrv:<D>`,
#                   `lcpp2775xmtp:<D>`) time one /completion request, because llama-bench drives no
#                   speculation; llama-server prints the draft acceptance and returns it in `timings`
#   LCPP27752_SRV_FLAGS, LCPP27754_SRV_FLAGS  the CLI flags with one slot (-np 1: the automatic slots
#                   would set four and a unified cache)
#   GLM_PREHEAT_K   the --n-cpu-moe K whose host set (tools/ref/gguf-ranges.py host) depth-glm5next.sh
#                   preheats before our arms and the fit twins: 45, every routed expert of blocks 3-44
#                   and token_embd. Blocks 3-44 are the engine's routed layers (blk.45, NextN, is no
#                   arm's host set but an MTP context's, whose leading-layer rule puts it on the card),
#                   so the set holds ours' host experts under any card count — the
#                   plan line's card_experts 39,941,832,704 B plus host_experts 145,536,581,632 B are
#                   exactly blocks 3-44's 185,478,414,336 — and a fit twin's trailing blocks. It is
#                   185.5 GB and token_embd of the 199.7 GB file, which the page cache holds whole
#   GLM_NCMOE_MTP   the MTP arm's --n-cpu-moe, GLM_NCMOE + 1: an MTP context loads the NextN block
#                   (blk.45, 4,378,853,376 B of experts and 200,306,816 B beside them), which the
#                   leading-layer rule keeps on the card, so one more block goes to the host: blocks
#                   37-45 hold the bytes 36-44 held, plus 191 MiB [derived]
#   GLM_MTP_FLAGS   the MTP draft: --spec-type draft-mtp --spec-draft-n-max 2 (#27754's body and the
#                   unsloth guide: n = 2 is its best, more drafts run slower)
#   LCPP27752_MTP_FLAGS, LCPP27754_MTP_FLAGS  the server flags at GLM_NCMOE_MTP with GLM_MTP_FLAGS
#   EXL3            the exllamav3 source tree; its bench is eval/perf.py, run by EXL3_PY (the venv of
#                   exllamav3 1.5.0 +cu128, torch 2.10.0), on EXL3_MODEL: another quantization (EXL3
#                   at EXL3_BPW bits a weight), so its rows are a table of their own, never a ratio
#   EXL3_FLAGS      perf.py's model flags: -mcs, the tail experts of every layer on the CPU with its
#                   dynamic hot/cold placement (its default), 224 of 288: 64 a layer on the A6000.
#                   ~~214 [derived from the two-card run of 09-15]~~ 214 does not load on the A6000
#                   alone ("Insufficient VRAM in split for model and cache", perf.py's default cache
#                   32768 and chunk 4096) and 224 does (load probe of round glmtime, 41 s);
#                   -mct 32 worker threads, as -t 32
#   EXL3_WIKITEXT   the wikitext-2 test text perf.py's token stream reads; its loader otherwise
#                   downloads it into the temp dir, so the runner stages this copy there first
#
# The MTP draft oracle (tools/ref/build-dump-mtp.sh, tools/ref/dump-mtp.sh; refset family
# mtp-glm5next). IK's tree has no MTP graph for this architecture, so the MTP set comes from another:
#   GLM_MTP_IK      ik's glm5next MTP graph merged onto upstream 7434a014, the tree whose llama-server
#                   measured the MTP accept rate on this box (rig-log 09-15). 7434a014 is an ancestor of
#                   IK's db517b69; the nine commits between them are V4.1's, the vision encoder's, the
#                   server's and iqk's (converters for the repacked quants, the flash-attention work
#                   buffer of a K with more than one head), none in a glm5next graph file
#   GLM_MTP_SHA     that tree's HEAD: the tree must still be at it, clean, and every MTP set names it as
#                   its `# build`; the refset family pins the same value (MTP_BUILD)
#
# Deliberately unset: IK_BEST_FLAGS, REF_PROMPTS and the ik and mistral.rs arms (IK_GPU_FLAGS, MRS,
# MRS_FLAGS): ik is the oracle, and mistral.rs (d5ae0f1 on the box) has no glm5next loader
# (glm4, glm4moe and glm4moelite only). Every script that reads them runs under `set -u`, so such a
# script stops at the unset name instead of running an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=glm5next
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_glm5next}"
: "${REF_SET_CUDA:=ref_cuda_glm5next}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads
# the vocabulary only). Five ids and no BOS: the file's pre-tokenizer is glm4, for which the
# vocabulary loader clears the BOS id, so the first id is 'The' and not tokenizer.ggml.bos_token_id.
# Changing them invalidates the whole set.
REF_TOKENS=785,6722,315,9621,374
REF_DUMP_LEASE=1
GLM_PROSE=glm5next/corpus-prose.ids
GLM_PROSE_SHA256=8af07981c1749170b57d424cff5274b89be063c1eb44ffa3a443ddc16642bb64
: "${GLM_PROSE_FROM:=50000}"
: "${LCPP27752:=/home/user/llama.cpp-pr27752}"
: "${LCPP27752BIN:=$LCPP27752/build/bin/llama-bench}"
: "${LCPP27754:=/home/user/llama.cpp-pr27754}"
: "${LCPP27754BIN:=$LCPP27754/build/bin/llama-bench}"
: "${LCPP27754_ENV=NVIDIA_TF32_OVERRIDE=0}"
__glm_dir=${BASH_SOURCE[0]%/*}
[ "$__glm_dir" != "${BASH_SOURCE[0]}" ] || __glm_dir=.
# shellcheck source=tools/ref/cards.sh
source "$__glm_dir/../cards.sh"
unset __glm_dir
# The 3090-vs-A6000 sizing below turns on GPU_3090; a timing card that is not the resolved
# A6000 with no 3090 UUID is undecidable — refuse rather than size the references for the
# wrong card (ref-paths.sh's refusal code).
if [ -n "${BLOOMERY_TIMING_GPU:-}" ] && [ -z "${GPU_3090:-}" ] && [ "$BLOOMERY_TIMING_GPU" != "${GPU_A6000:-}" ]; then
  echo "models/glm5next.sh: BLOOMERY_TIMING_GPU=$BLOOMERY_TIMING_GPU, and tools/ref/cards.sh resolved no 3090 UUID (${CARDS_ERROR:-no reason given}): whether the timing card is the 3090 sizes GLM_NCMOE (42 vs 36)" >&2
  exit 64
fi
if [ -n "${BLOOMERY_TIMING_GPU:-}" ] && [ "$BLOOMERY_TIMING_GPU" = "${GPU_3090:-}" ]; then
  : "${GLM_NCMOE:=42}"
else
  : "${GLM_NCMOE:=36}"
fi
: "${LCPP27752_GPU_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE -fa on -t 32 -nopo 1}"
: "${LCPP27754_GPU_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE -fa off -t 32 -nopo 1}"
: "${LCPP27752_CLI_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE -fa on -t 32 --no-op-offload -fit off}"
: "${LCPP27754_CLI_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE -fa off -t 32 --no-op-offload -fit off}"
: "${LCPP27752SRV:=$LCPP27752/build/bin/llama-server}"
: "${LCPP27754SRV:=$LCPP27754/build/bin/llama-server}"
: "${GLM_NCMOE_MTP:=$((GLM_NCMOE + 1))}"
: "${GLM_PREHEAT_K:=45}"
: "${LCPP27752_SRV_FLAGS:=$LCPP27752_CLI_FLAGS -np 1}"
: "${LCPP27754_SRV_FLAGS:=$LCPP27754_CLI_FLAGS -np 1}"
: "${GLM_MTP_FLAGS:=--spec-type draft-mtp --spec-draft-n-max 2}"
: "${LCPP27752_MTP_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE_MTP -fa on -t 32 --no-op-offload -fit off -np 1 $GLM_MTP_FLAGS}"
: "${LCPP27754_MTP_FLAGS:=-ngl 999 --n-cpu-moe $GLM_NCMOE_MTP -fa off -t 32 --no-op-offload -fit off -np 1 $GLM_MTP_FLAGS}"
: "${EXL3:=/home/user/exllamav3-src}"
: "${EXL3_PY:=/home/user/.venv-exl3/bin/python}"
: "${EXL3_MODEL:=/models/GLM-5.3-Flash-exl3-4.05}"
: "${EXL3_BPW:=4.05}"
: "${EXL3_FLAGS:=-mcs 224 -mct 32}"
: "${EXL3_WIKITEXT:=/home/user/eval/wikitext-2-raw/wiki.test.raw}"
: "${GLM_MTP_IK:=/home/user/ik-glm53-mtp}"
GLM_MTP_SHA=425a2c1d

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-glm5next <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches and the recurrent state the step reads carry a dumped
# prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048
#   d3kdsa   ik's k-pool indexer on (--dsa): a quiet prefill of 3,070 ids of prose, the step at
#            position 3,070, -c 4096 — past the 2,051 positions a latent layer keeps whole, so the
#            step's latent layers attend the 512 pools the indexer picks and their tail
#   d16kdsa  the same at a quiet prefill of 16,382, the step at position 16,382, -c 16384: the
#            deepest context the program serves (place::ORACLE_POSITIONS)
#   Both --dsa steps sit at (q + 1) % 4 == 3, so the step's tail is three real cells and ik's
#   zero-filled tail slots (src/llama.cpp, inp_kpool_tail) name no cell at the step; its prefill
#   rows past 2,051 with a shorter tail still list cell 0 there, which the model and mainline do
#   not (build_glm5next.cpp; llama.cpp #27752 pads with a masked cell).
# The prose is $BLOOMERY_DATA/glm5next/corpus-prose.ids: the tokenizer oracle's `prose.nps.ids` under
# this vocabulary (set tokenizer-glm5next, written by crates/tokenizer/tools/oracle.sh with
# TOKENIZER_VOCAB = MODEL and ik-idxkey's llama-tokenize; the text is ik's docs/**/*.md and
# README.md, md5 c6bb074439479420faffa33257424c00, the prose the qwen3moe and qwen35moe sets are of),
# one id per line, 71,727 ids, copied out of tokenizer-glm5next/ so a rerun of that oracle on a moved
# ik tree cannot change the ids a set is of; the sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_glm5next_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_glm5next_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    d3kdsa)  STEP_SET=ref_glm5next_d3kdsa;  STEP_CTX=4096;  STEP_PREFILL=3070;  STEP_ARGS+=(--dsa) ;;
    d16kdsa) STEP_SET=ref_glm5next_d16kdsa; STEP_CTX=16384; STEP_PREFILL=16382; STEP_ARGS+=(--dsa) ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|d3kdsa|d16kdsa) STEP_TOKENS_FILE=$BLOOMERY_DATA/$GLM_PROSE
                        STEP_TOKENS_SHA256=$GLM_PROSE_SHA256 ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
