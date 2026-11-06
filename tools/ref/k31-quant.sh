#!/usr/bin/env bash
# k31-quant.sh — the Kimi K3.1 Pro low-bit GGUF pipeline's runner: prepare | imatrix | quantize | splice | all | upload, and the smoke.
#
#   BLOOMERY_MODEL=kimi-k3 tools/box.sh 'bash tools/ref/k31-quant.sh <step> [options]'          (just k31-quant <step> [options])
#
# The pipeline (round k31quant; the lead runs the big steps). Each step is the runner's; none takes the timing lease or a hold: the
# pipeline is not a measurement and runs hours, so the lead schedules timed runs around it. The runner's own exclusion is the lock
# $K3Q_WORK/run.lock (flock -n; a second run is refused by name, rc 75), imatrix and quantize run under nice -n 10 / ionice -c2 -n7,
# and every step's log records its wall and bytes. Only the smoke can take the CPU lease (`--lease`: it measures the quantizer's rate).
#   prepare   the `prepare` binary of round k31hf, twice: the HF folder -> a GGUF in K3's shape (`--drop 'blk.*.indexer.*' --arch
#             kimi-k3`: the experts as MXFP4 blocks, no indexer tensors) and -> the indexer-only GGUF (`--only 'blk.*.indexer.*'`).
#             The built prepare is refused by name when its usage lacks --drop, --only or --arch.
# Then, each a step of this runner:
#   imatrix   llama-imatrix (mainline 78feec062, built by tools/ref/build-k3-quant-tree.sh as its own tree) over a calibration text
#             at ctx 2048 -> $K3Q_WORK/kimi-k31-c<ctx>x<chunks>.imatrix.gguf. The text is the K3 trace corpora's own sources (the engine
#             tree's docs and README, its C++ sources), half the chunks of each, cut by the K3 tokenizer's own bytes per token.
#   quantize  llama-quantize --allow-requantize --imatrix <that> --tensor-type-file <rules> [--keep-split] <in> <out> <ftype>, the map
#             --map says (below), after llama-quantize's own --dry-run plan of every tensor's type and size over the input's headers
#             (kept in $K3Q_WORK; the output is held to it tensor by tensor afterwards)
#             -> $K3Q_OUT_BASE/Kimi-K3.1-Pro-<type>[-<map>]-k3shape/ (a K3-shaped set: mainline loads it as K3, which at ctx <= 2048 is K3.1).
#             Maps (what the lead's placement sizes): `tight` (default) routed experts IQ1_S, the trunk Q4_K_M (the global ftype; attn_k_b
#             and ssm_f_b Q8_0, ssm_beta F32, kept for their size and sensitivity), token_embd Q4_K, output Q6_K, norms F32;
#             `trunk-q5` the same with a Q5_K_M trunk; `ud` the K3 UD-IQ1_S file's own table (experts IQ1_S with IQ2_XXS layer bumps, the
#             trunk Q6_K), reproduced by rules (`rules` prints them, `check` proves them against the K3 table): the comparison.
#   plan      the quantize step's plan alone: no lease, no write besides the plan files; prints the bytes per class and the totals.
#   splice    tools/ref/gguf-splice.py add: the indexer tensors back into that set, `general.architecture` and the `kimi-k3.*` keys
#             renamed to $K3Q_ARCH, the dsa_config keys added; the shards that do not change are hard links; the source LICENSE copied
#             in and the model card README.md filled from tools/ref/k31-model-card.md.in
#             -> $K3Q_OUT_BASE/Kimi-K3.1-Pro-<type>[-<map>]/.
#   upload    that directory to the Hub: `uvx --from huggingface_hub hf upload-large-folder <repo> <dir> --repo-type model` (resumable by
#             itself), refused by name without the splice step's `.done` marker, a README.md or a LICENSE; never prints the token.
#   all       the three in that order (each its own lease and card).   rules, check: the type map; no box, no lease.
#   calib     build the imatrix's calibration text only (the tokenizer loads the vocabulary of --in, no tensor): no lease, one thread
#             for the length of the tokenization; the imatrix step builds the same text again.
#   smoke     the whole chain on the K3 fixture (152 MB, K3's vocabulary and shapes), under one CPU lease with --lease: a K3.1 stand-in with MXFP4 experts,
#             a split set with a synthetic indexer, strip, imatrix (2 chunks, ctx 512), quantize (1 thread: the quantizer's rate),
#             splice with --arch kimi-k31, and a read back. Its numbers are the smoke's, not claims.
#
# Options (an option's environment variable is in brackets):
#   --in FILE          the K3-shaped set's first shard, the strip's output      [K3Q_IN]  (imatrix, quantize, all: required, but --dry-run: the
#                      K3 GGUF's; after `prepare` they find its output; splice: the quantized set's; upload: the splice's output DIR)
#   --map M            tight (default) | trunk-q5 | ud: the type map (above)    [K3Q_MAP]
#   --type T           IQ1_S (default) | IQ2_XXS: the routed experts' type (ud: IQ2_XXS uniformly, unless --bump-* names layers)
#   --ctx N --chunks N the imatrix's context (2048) and chunk count (4); smoke 512 and 2
#   --threads N        llama-imatrix and llama-quantize threads                 [K3Q_THREADS]  (default: nproc)
#   --corpus FILE      a calibration text of your own                           [K3Q_CORPUS]
#   --ref-table TSV    the reference table: name TAB type TAB dims (default crates/model/tests/kimi_k3_tensors.tsv, K3's)
#   --bump-gateup L|L  --bump-down L|L   layers whose experts get the bump type (IQ2_XXS); IQ1_S derives them from the table
#   --indexer FILE     the indexer tensors' GGUF (prepare's `--only` output)    [K3Q_INDEXER]
#   --dsa-config FILE  the K3.1 config.json (text_config.dsa_config) -> the --kv keys of the splice     [K3Q_DSA_CONFIG]
#   --arch NAME        the final architecture word (default kimi-k31)           [K3Q_ARCH]
#   --out-base DIR     where the sets go (default /models)                      [K3Q_OUT_BASE]
#   --force            remove the step's outputs and done marker by name, and run it again
#   --dry-run          print every command with its card, bound and the derived wall; take no lease, write nothing, read headers only
#   --detach           start the runner under setsid and return: pid, log and rc file are under $K3Q_WORK (poll `kill -0 <pid>`)
#   --pred-s           print the predicted wall in whole seconds (no file is read) and exit: the box queue's BLOOMERY_Q_PRED_S
#   --lease            smoke only: take the CPU lease with the smoke's card (the 1-thread rate is a measurement)
#   --max-trunk-gb N   quantize: refuse when the plan's trunk class exceeds N GB (default 35 for tight; 0 = off)
#   --hf-folder DIR    the K3.1 HF folder: prepare's input, the LICENSE and the source revision   [K3Q_HF_FOLDER] (/data/models/Kimi-K3.1-Pro)
#   --license FILE     splice: the LICENSE copied into the output (default: <hf-folder>/LICENSE)
#   --source-rev REV   splice: the HF commit the folder holds, for the card (default: read from the folder's own download metadata)
#   --prepare-bin BIN  prepare: the built `prepare` (default: target/release/prepare); --follow LIST, --shard-gib N pass through to it
#   --repo OWNER/NAME  upload: the Hub repo (default midagedev/Kimi-K3.1-Pro-GGUF)
# Environment: K3Q_TOOLS_COMMIT (the bloomery commit the model card names for these tools; the box's tree has no .git), K3Q_TREE (default /home/user/llama.cpp-kimik3-quant), K3Q_WORK (default /root/k31quant), K3Q_FIXTURE (the smoke's file),
#   K3Q_BOUND_S (a step's timeout in seconds; default three times its predicted wall, at least 600), K3Q_UPLOAD_BOUND_S (default 86400).
#
# Resumable: a step whose output is complete and whose `.done` marker (the identity of its input shards by name, size and mtime,
# the tool's sha256, the flags) matches is skipped; a marker that no longer matches is refused (rerun with --force); a partial output
# (`<out>.partial`) is removed by name at the start of the next run. A step is not resumable inside: llama-imatrix and llama-quantize
# start over. Every command runs under `timeout` through lease_bounded, its pid in $K3Q_WORK/<step>.pid, its output in
# $K3Q_WORK/<step>.log. The cards (docs/cards/k31-*.card) state each step's prediction; only the smoke's is a lease card.
#
# The big files (the K3.1 HF folder, the K3 and K3.1 GGUF sets) are never read in full here: this script reads their headers
# (gguf-splice.py) and their sizes (stat); the tools it starts read them, and that is the step.
#
# Exit: 0 done or skipped by its marker; 64 a usage error or a refusal by name; 65 an input or output that is not what the step
# needs; 66 a missing file or tool; 70 the runner's own failure; 75 the run lock (or the smoke's lease) is held: contention;
# 124 a bound; a tool's own code otherwise (the step's log has its output).
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SELF=$HERE/$(basename "${BASH_SOURCE[0]}")
SPLICE=$HERE/gguf-splice.py
ORIG_ARGS=("$@")

# ---- the derivation: the terms and where each came from (a number measured is named so; the rest is [derived]) ----
K3_BYTES=594030000000       # the K3 UD-IQ1_S set: 14 shards, `ls -l` on the box (594.03 GB)
K3_CHUNK_S=546.5            # measured: k3trace's sizing chunk on that set, one 2048-token call (majflt 4.8 M, window read ~1.13 TB)
K3_POPULATE_S=367.6         # measured: k3trace's load_s on that set (MAP_POPULATE of the whole file; 368-429 s over its runs)
K31_BYTES=1560000000000     # derived: the K3.1 GGUF from prepare: 1.56 TB (MXFP4 experts + the trunk), the k31 report's sizes
K31_EXPERT_W=2722740000000  # derived from shapes: the 276 routed-expert tensors of K3's table, 3584 x 3072 x 896 each (K3.1 has K3's)
NVME_READ_GBS=1.6           # measured floor: /models page-fault reads, 1.0-1.6 GB/s (k3trace); the sequential read uses the top
NVME_WRITE_GBS=1.5          # derived: the same drive's write, no measurement better than the read's
OUT_PER_IN=0.362            # derived: llama-quantize --dry-run of the tight map over K3's shapes: 567.7 GB out of 1.56 TB in
QUANT_RATE_MW=5.4           # measured: the smoke's 1-thread rate on the fixture (5.36 Mweights/s/thread, 4.72 Mweights in 881 ms, I/O included);
                            # the 40-thread scaling below is derived, not measured
QUANT_EFF_THREADS=40        # derived: 64 hardware threads on 32 cores, the quantizer's row loop scales ~0.6 on the SMT siblings
SPLICE_SHARD_BYTES=48000000000   # the receiving shard's size, a K3 UD shard's (47-52 GB)

die() { echo "k31-quant.sh: $1" >&2; exit "${2:-64}"; }
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}" >&2; exit 64; }
say() { echo "k31-quant.sh: $*"; }

# ---- options ----
[ $# -ge 1 ] || usage
case $1 in -h | --help) usage ;; esac
STEP=$1
shift
DRY=0 SMOKE=0 DETACH=0 FORCE=0 PRED=0 USE_LEASE=0
MAP=${K3Q_MAP:-tight} REPO=midagedev/Kimi-K3.1-Pro-GGUF HF=${K3Q_HF_FOLDER:-/data/models/Kimi-K3.1-Pro} LICENSE_F='' SRC_REV='' PREPARE_BIN='' FOLLOW='' SHARD_GIB='' MAX_TRUNK=''
TYPE=IQ1_S CTX='' CHUNKS='' IN=${K3Q_IN:-} CORPUS=${K3Q_CORPUS:-} INDEXER=${K3Q_INDEXER:-} DSA=${K3Q_DSA_CONFIG:-}
ARCH=${K3Q_ARCH:-kimi-k31} THREADS=${K3Q_THREADS:-} REF='' OUT_BASE=${K3Q_OUT_BASE:-/models} BUMP_GU='' BUMP_DN='' BUMP_SET=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --detach) DETACH=1 ;;
    --force) FORCE=1 ;;
    --pred-s) PRED=1 ;;
    --smoke) SMOKE=1 ;;
    --lease) USE_LEASE=1 ;;
    --map | --repo | --hf-folder | --license | --source-rev | --prepare-bin | --follow | --shard-gib | --max-trunk-gb)
      [ $# -ge 2 ] || die "$1 takes a value"
      case $1 in
        --map) MAP=$2 ;; --repo) REPO=$2 ;; --hf-folder) HF=$2 ;; --license) LICENSE_F=$2 ;; --source-rev) SRC_REV=$2 ;;
        --prepare-bin) PREPARE_BIN=$2 ;; --follow) FOLLOW=$2 ;; --shard-gib) SHARD_GIB=$2 ;; --max-trunk-gb) MAX_TRUNK=$2 ;;
      esac
      shift ;;
    --in | --type | --ctx | --chunks | --threads | --corpus | --ref-table | --bump-gateup | --bump-down | --indexer | --dsa-config | --arch | --out-base)
      [ $# -ge 2 ] || die "$1 takes a value"
      case $1 in
        --in) IN=$2 ;; --type) TYPE=$2 ;; --ctx) CTX=$2 ;; --chunks) CHUNKS=$2 ;; --threads) THREADS=$2 ;; --corpus) CORPUS=$2 ;;
        --ref-table) REF=$2 ;; --bump-gateup) BUMP_GU=$2 BUMP_SET=1 ;; --bump-down) BUMP_DN=$2 BUMP_SET=1 ;;
        --indexer) INDEXER=$2 ;; --dsa-config) DSA=$2 ;; --arch) ARCH=$2 ;; --out-base) OUT_BASE=$2 ;;
      esac
      shift ;;
    -h | --help) usage ;;
    *) die "unknown option '$1' (k31-quant.sh --help)" ;;
  esac
  shift
done
case $STEP in
  imatrix | quantize | splice | all | rules | check | calib | plan | prepare | upload) ;;
  smoke) SMOKE=1 ;;
  *) die "step '$STEP' is not prepare, imatrix, quantize, plan, splice, all, upload, rules, check, calib or smoke" ;;
esac
case $MAP in
  tight) QTYPE=Q4_K_M TOKEMB=q4_k MAPTAG='' ;;
  trunk-q5) QTYPE=Q5_K_M TOKEMB=q4_k MAPTAG=-trunk-q5 ;;
  ud) QTYPE='' TOKEMB=q6_k MAPTAG=-ud ;;
  *) die "--map $MAP: tight, trunk-q5 or ud" ;;
esac
[ -n "$MAX_TRUNK" ] || { if [ "$MAP" = tight ]; then MAX_TRUNK=35; else MAX_TRUNK=0; fi; }
[ "$USE_LEASE" = 0 ] || [ "$SMOKE" = 1 ] || die "--lease is the smoke's alone: the other steps take no lease (the lead schedules timed runs around them)"
case ${MAX_TRUNK:-0} in '' | *[!0-9]*) die "--max-trunk-gb wants whole GB (0 = off), got '$MAX_TRUNK'" ;; esac
case $REPO in */* ) case $REPO in */*/* | /* | */) die "--repo $REPO: OWNER/NAME" ;; esac ;; *) die "--repo $REPO: OWNER/NAME" ;; esac
[ -z "$SHARD_GIB" ] || case $SHARD_GIB in '' | *[!0-9]* | 0) die "--shard-gib wants whole GiB above 0, got '$SHARD_GIB'" ;; esac
case $TYPE in IQ1_S | IQ2_XXS) ;; *) die "--type $TYPE: IQ1_S or IQ2_XXS (the two the pipeline's cards are written for)" ;; esac
if [ "$SMOKE" = 1 ]; then CTX=${CTX:-512} CHUNKS=${CHUNKS:-2}; else CTX=${CTX:-2048} CHUNKS=${CHUNKS:-4}; fi
for v in CTX CHUNKS; do
  case ${!v} in '' | *[!0-9]* | 0) die "--$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]') wants a whole number above 0, got '${!v}'" ;; esac
done
[ "$DRY" = 1 ] && [ "$DETACH" = 1 ] && die "--detach and --dry-run do not combine"
ARCH_OK=$(printf '%s' "$ARCH" | tr -d 'a-z0-9-')
[ -z "$ARCH_OK" ] && [ -n "$ARCH" ] || die "--arch $ARCH: letters, digits and '-' only (it names every key prefix)"

TREE=${K3Q_TREE:-/home/user/llama.cpp-kimik3-quant}
WORK_BASE=${K3Q_WORK:-/root/k31quant}
WORK=$WORK_BASE
[ "$SMOKE" = 1 ] && WORK=$WORK_BASE/smoke
LOGDIR=$WORK
TAG=$TYPE$MAPTAG
[ "$SMOKE" = 1 ] && TAG=smoke
Q=$TREE/build/bin/llama-quantize
IMAT=$TREE/build/bin/llama-imatrix
TOKENIZE=$TREE/build/bin/llama-tokenize
GSPLIT=$TREE/build/bin/llama-gguf-split
[ "$SMOKE" = 1 ] && OUT_BASE=$WORK/out
[ -n "$REF" ] || REF=$ROOT/crates/model/tests/kimi_k3_tensors.tsv
# Checked here, not in bound_of: a die inside $(...) ends only the substitution's subshell, and the step would start with no bound.
case ${K3Q_BOUND_S-x} in x) ;; '' | *[!0-9]* | 0) die "K3Q_BOUND_S wants whole seconds above 0, got '${K3Q_BOUND_S}'" ;; esac

# ---- shard-set helpers (names, sizes: no data is read) ----
shard_count() {
  case $1 in
    *-00001-of-[0-9][0-9][0-9][0-9][0-9].gguf) local f=${1##*-of-}; echo $((10#${f%.gguf})) ;;
    *) echo 1 ;;
  esac
}
shard_path() { # <first shard> <1-based index>
  local n
  n=$(shard_count "$1")
  if [ "$n" = 1 ]; then echo "$1"; else printf '%s-%05d-of-%05d.gguf\n' "${1%-00001-of-*}" "$2" "$n"; fi
}
set_ident() { # "path size mtime" per shard; a missing shard is refused by name
  local n i p
  n=$(shard_count "$1")
  for ((i = 1; i <= n; i++)); do
    p=$(shard_path "$1" "$i")
    [ -f "$p" ] || die "shard $i of $n of the set $1 is missing: $p" 66
    stat -c '%n %s %Y' "$p"
  done
}
set_bytes() { set_ident "$1" | awk '{ s += $(NF-1) } END { printf "%d", s }'; }
named_first() { # <a set's first shard> <dir> <prefix>: the same shard count under another name
  local n
  n=$(shard_count "$1")
  if [ "$n" = 1 ]; then echo "$2/$3.gguf"; else printf '%s/%s-00001-of-%05d.gguf\n' "$2" "$3" "$n"; fi
}
first_match() { # the first argument that is a file (a glob left unexpanded is none)
  local f
  for f in "$@"; do
    if [ -f "$f" ]; then
      echo "$f"
      return 0
    fi
  done
  return 1
}
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
key_of() { printf '%s\n' "$@" | sha256sum | cut -d' ' -f1; }
splice() { python3 -I "$SPLICE" "$@"; }

# ---- the derived walls (seconds; every number above is a term here) ----
pred_imatrix() { # <bytes> <chunks>: populate + chunks x one chunk, both scaled by bytes from the measured K3 set
  awk -v b="$1" -v n="$2" -v kb="$K3_BYTES" -v c="$K3_CHUNK_S" -v p="$K3_POPULATE_S" 'BEGIN { printf "%d", (p + n * c) * b / kb + 0.5 }'
}
pred_quantize() { # <bytes> <threads>: read + quantizer CPU + write, in sequence (the quantizer reads a tensor, quantizes it, writes it)
  awk -v b="$1" -v e="$QUANT_EFF_THREADS" -v r="$QUANT_RATE_MW" -v rd="$NVME_READ_GBS" -v wr="$NVME_WRITE_GBS" -v o="$OUT_PER_IN" \
    -v w="$K31_EXPERT_W" -v kb="$K31_BYTES" 'BEGIN { printf "%d", (w * b / kb) / (e * r * 1e6) + b / (rd * 1e9) + b * o / (wr * 1e9) + 0.5 }'
}
pred_splice() { # the receiving shard read, written and read back, at the drive's rates, and the first shard
  awk -v s="$SPLICE_SHARD_BYTES" -v rd="$NVME_READ_GBS" -v wr="$NVME_WRITE_GBS" 'BEGIN { printf "%d", s / (rd * 1e9) * 2 + s / (wr * 1e9) + 5 + 0.5 }'
}
pred_of() { # <step> <bytes>
  case $1 in
    imatrix) pred_imatrix "$2" "$CHUNKS" ;;
    quantize) pred_quantize "$2" ;;
    splice) pred_splice ;;
  esac
}
bytes_assumed() { [ "$SMOKE" = 1 ] && echo 152373440 || echo "$K31_BYTES"; }
bound_of() { # <predicted seconds>
  if [ -n "${K3Q_BOUND_S:-}" ]; then
    echo "$K3Q_BOUND_S"
  else
    awk -v p="$1" 'BEGIN { b = 3 * p; if (b < 600) b = 600; printf "%d", b }'
  fi
}
hms() { awk -v s="$1" 'BEGIN { printf "%d s = %dh%02dm", s, s / 3600, (s % 3600) / 60 }'; }

if [ "$PRED" = 1 ]; then # no file is read: the projection for the 1.56 TB set (the smoke's own for --smoke)
  if [ "$SMOKE" = 1 ]; then echo 900; exit 0; fi
  case $STEP in
    imatrix | quantize | splice) pred_of "$STEP" "$K31_BYTES" ;;
    all) echo $(($(pred_of imatrix "$K31_BYTES") + $(pred_of quantize "$K31_BYTES") + $(pred_of splice "$K31_BYTES"))) ;;
    *) echo 60 ;;
  esac
  exit 0
fi

# ---- the type map: llama-quantize --tensor-type-file rules, K3 UD-IQ1_S's ----
# The file is whitespace-separated `regex=type` tokens (llama-quantize reads it with `file >> arg`: no comments, no spaces); the
# first rule whose regex (std::regex_search, the name lowercased) matches wins, and the rules are consulted only for tensors the
# quantizer touches (2-D or more, `weight`, not a norm, not ffn_gate_inp, not ssm_conv1d) and only when the global type is itself a
# quantized one. So: the layer bumps first, then every routed expert, then the stems the K3 file holds as Q8_0 and as F32 (ssm_beta
# is a 2-D weight the quantizer would otherwise quantize), then the Q6_K stems. token_embd and output.weight go by the two flags.
gen_rules() { # <expert type> <gate/up bump layers 'a|b'> <down bump layers> <bump type>
  [ -z "$2" ] || printf '%s\n' "blk\\.($2)\\.ffn_(gate|up)_exps\\.weight=$4"
  [ -z "$3" ] || printf '%s\n' "blk\\.($3)\\.ffn_down_exps\\.weight=$4"
  printf '%s\n' "ffn_(down|gate|up)_exps\\.weight=$1" \
    '\.attn_k_b\.weight$=q8_0' '\.ssm_f_b\.weight$=q8_0' '\.ssm_beta\.weight$=f32' \
    '\.(attn_gate|attn_k|attn_kv_a_mqa|attn_output|attn_q|attn_q_a|attn_q_b|attn_v|attn_v_b|ffn_down|ffn_gate|ffn_up|ffn_down_shexp|ffn_gate_shexp|ffn_up_shexp|ffn_routed_down|ffn_routed_up|ssm_f_a|ssm_g)\.weight$=q6_k' \
    '^(token_embd|output)\.weight$=q6_k'
}
gen_rules_trunk() { # <expert type>: the routed experts, then the three small exceptions; every other tensor takes the global type's
  printf '%s\n' "ffn_(down|gate|up)_exps\\.weight=$1" '\.attn_k_b\.weight$=q8_0' '\.ssm_f_b\.weight$=q8_0' '\.ssm_beta\.weight$=f32'
}
ref_layers() { # <gate|up|down> <type>: the layers of $REF whose routed tensor of that kind has that type, as 'a|b|c'
  awk -F'\t' -v kind="$1" -v ty="$2" '$2 == ty && $1 ~ ("^blk\\.[0-9]+\\.ffn_" kind "_exps\\.weight$") { split($1, a, "."); print a[2] }' "$REF" |
    sort -un | paste -sd'|' -
}
derive_bumps() { # sets BUMP_GU and BUMP_DN: IQ1_S follows the table (K3 UD's choice); another type only what --bump-* names
  [ "$MAP" = ud ] || return 0
  [ "$BUMP_SET" = 0 ] || return 0
  [ "$TYPE" = IQ1_S ] || return 0
  [ -f "$REF" ] || die "no reference table at $REF (--ref-table)" 66
  local g u
  g=$(ref_layers gate IQ2_XXS)
  u=$(ref_layers up IQ2_XXS)
  [ "$g" = "$u" ] || die "$REF: the layers whose gate experts are IQ2_XXS ($g) are not the up experts' ($u): the rule set has one list for both" 65
  BUMP_GU=$g
  BUMP_DN=$(ref_layers down IQ2_XXS)
}
make_rules() { # <file>
  case $MAP in
    ud)
      derive_bumps
      gen_rules "$TYPE" "$BUMP_GU" "$BUMP_DN" IQ2_XXS > "$1" ;;
    *) gen_rules_trunk "$TYPE" > "$1" ;;
  esac
}

# ---- the profile, the lease, one bounded command ----
LEASE_HELD=0 PROFILE_LOADED=0 PLANNED=0
load_profile() {
  [ "$PROFILE_LOADED" = 0 ] || return 0
  BLOOMERY_MODEL=${BLOOMERY_MODEL:-kimi-k3}
  export BLOOMERY_MODEL
  # shellcheck source=tools/ref/ref-paths.sh
  source "$HERE/ref-paths.sh" || die "ref-paths.sh failed" 70
  [ "$MODEL_NAME" = kimi-k3 ] || die "the profile is $MODEL_NAME, not kimi-k3: BLOOMERY_MODEL=kimi-k3 tools/box.sh '...'" 64
  PROFILE_LOADED=1
}
load_lease() {
  [ "$DRY" = 0 ] || return 0
  if [ "$USE_LEASE" = 0 ]; then # no lease: only the bounded runner of the gates is needed
    # shellcheck source=tools/gate-bound.sh
    source "$ROOT/tools/gate-bound.sh" || die "gate-bound.sh failed" 70
    return 0
  fi
  command -v flock > /dev/null 2>&1 || die "no flock(1): the lease cannot be taken on this host (the box has it)" 70
  # shellcheck source=tools/ref/lease.sh
  source "$HERE/lease.sh" || die "lease.sh failed" 70
  WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model busiest)
}
NOLEASE_SAID=0
lease_on() { # <card slug>: the lease for the rest of this run (a smoke --lease holds one for all its steps); without --lease, none
  if [ "$USE_LEASE" = 0 ]; then
    [ "$NOLEASE_SAID" = 1 ] || say "$([ "$DRY" = 1 ] && echo '[dry-run] ')no timing lease and no hold (the card docs/cards/$1.card states the prediction); run lock $WORK_BASE/run.lock; imatrix and quantize under nice -n 10, ionice -c2 -n7"
    NOLEASE_SAID=1
    return 0
  fi
  if [ "$DRY" = 1 ]; then
    [ "$PLANNED" = 1 ] || say "[dry-run] lease_take with BLOOMERY_LEASE_CARD=docs/cards/$1.card (card.py lease checks it first)"
    PLANNED=1
    return 0
  fi
  [ "$LEASE_HELD" = 0 ] || return 0
  BLOOMERY_LEASE_CARD=docs/cards/$1.card
  export BLOOMERY_LEASE_CARD
  lease_take
  LEASE_HELD=1
  witness "pre-$STEP"
}
lease_off() {
  [ "$DRY" = 0 ] && [ "$LEASE_HELD" = 1 ] && [ "${KEEP_LEASE:-0}" = 0 ] || return 0
  witness "post-$STEP"
  lease_release
  LEASE_HELD=0
}
run() { # <name> <bound s> <command...>: the command under a bound; output in $LOGDIR/<name>.log; returns its status
  local name=$1 bound=$2 rc=0 t0
  shift 2
  if [ "$DRY" = 1 ]; then
    printf '  [dry-run] %s: timeout %ss, log %s/%s.log\n      ' "$name" "$bound" "$LOGDIR" "$name"
    printf '%q ' "$@"
    echo
    return 0
  fi
  printf '%q ' "$@" > "$LOGDIR/$name.cmd"
  : > "$LOGDIR/$name.log"
  say "$name: started $(date -u +%H:%M:%S), bound ${bound}s, log $LOGDIR/$name.log"
  t0=$SECONDS
  LEASE_BOUNDED_OUT=$LOGDIR/$name.log LEASE_BOUNDED_PIDFILE=$LOGDIR/$name.pid bounded "$bound" "$@" || rc=$?
  rm -f "$LOGDIR/$name.pid"
  say "$name: ended $(date -u +%H:%M:%S), rc $rc, wall $((SECONDS - t0)) s"
  return "$rc"
}
acquire_lock() { # the runner's own mutual exclusion: one k31-quant.sh run at a time (children inherit fd 8 and keep it while they live)
  command -v flock > /dev/null 2>&1 || die "no flock(1) on this host" 70
  mkdir -p "$WORK_BASE"
  exec 8>> "$WORK_BASE/run.lock"
  flock -n 8 || die "another k31-quant.sh run holds $WORK_BASE/run.lock ($(tr '\n' ' ' < "$WORK_BASE/run.lock" 2> /dev/null)): wait for it, or look at its pid" 75
  printf 'pid %s step %s since %s\n' "$$" "$STEP" "$(date -u +%FT%TZ)" > "$WORK_BASE/run.lock"
}
fail_step() { # <name> <rc>: the log's tail, then the exit
  echo "k31-quant.sh: $1 failed with rc $2; the last lines of $LOGDIR/$1.log:" >&2
  tail -n 15 "$LOGDIR/$1.log" 2> /dev/null | sed 's/^/    /' >&2
  [ "$2" != 124 ] || echo "k31-quant.sh: rc 124: the bound cut it off; K3Q_BOUND_S moves it" >&2
  exit "$2"
}
partial_clean() { # <path ending .partial>: removed by name, only under the work or out directories
  case $1 in *.partial) ;; *) die "refusing to remove '$1': not a .partial path" 70 ;; esac
  [ -e "$1" ] || return 0
  case $1 in "$WORK"/* | "$OUT_BASE"/*) ;; *) die "refusing to remove '$1': outside $WORK and $OUT_BASE" 70 ;; esac
  if [ "$DRY" = 1 ]; then say "[dry-run] would remove the partial output $1"; return 0; fi
  say "removing the partial output of an earlier run: $1"
  rm -rf -- "$1"
}
marker_state() { # <marker> <key> <output path>: 0 done and matching, 1 absent; refuses a stale one by name
  [ -f "$1" ] || return 1
  [ "$(sed -n 's/^key=//p' "$1")" = "$2" ] ||
    die "$1 records another input or flags than this run's (its key $(sed -n 's/^key=//p' "$1" | cut -c1-12), now ${2:0:12}): --force runs the step again" 65
  [ -e "$3" ] || die "$1 says the step is done, but its output $3 is gone: --force runs the step again" 65
  return 0
}
write_marker() { # <marker> <key> <output> <note>
  printf 'key=%s\nout=%s\nat=%s\n%s\n' "$2" "$3" "$(date -u +%FT%TZ)" "$4" > "$1"
}
force_clean() { # <marker> <output...>
  [ "$FORCE" = 1 ] || return 0
  local m=$1 o
  shift
  [ "$DRY" = 0 ] || { say "[dry-run] --force would remove $m and $*"; return 0; }
  rm -f -- "$m"
  for o in "$@"; do
    case $o in "$WORK"/* | "$OUT_BASE"/*) rm -rf -- "$o" ;; *) die "--force: refusing to remove '$o': outside $WORK and $OUT_BASE" 70 ;; esac
  done
}

# ---- the calibration text ----
tok_count() { # <model's first shard> <text file>: the token count of the K3 tokenizer (vocab only: no tensor is read)
  CUDA_VISIBLE_DEVICES='' "$TOKENIZE" -m "$1" -f "$2" --ids --no-escape --no-parse-special --log-disable | tr -d '[] ' | tr ',' '\n' | grep -c .
}
gather() { # <prose|code>: the K3 trace corpora's sources (tools/ref/kimi-k3-corpus.sh), in its order
  case $1 in
    code) find "$IK/src" -maxdepth 1 -name '*.cpp' | sort | xargs -d '\n' cat -- ;;
    prose) { find "$IK/docs" -name '*.md' | sort | xargs -d '\n' cat --; cat -- "$IK/README.md"; } ;;
  esac
}
build_calib() { # <out> <in's first shard>: half the chunks of prose, then code, cut at the bytes the tokenizer says (bytes per token)
  local out=$1 in=$2 tmp half need pt ct pb cb want total
  [ -x "$TOKENIZE" ] || die "no llama-tokenize at $TOKENIZE: just build-k3-quant-tree" 66
  [ -d "$IK/docs" ] && [ -d "$IK/src" ] || die "no text sources under IK=$IK (docs/, src/): the calibration text is the K3 trace corpora's" 66
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/k31calib.XXXXXX") || die "no temporary directory" 70
  gather prose > "$tmp/prose.txt"
  gather code > "$tmp/code.txt"
  half=$(((CHUNKS + 1) / 2))
  need=$((half * CTX))
  pt=$(tok_count "$in" "$tmp/prose.txt")
  ct=$(tok_count "$in" "$tmp/code.txt")
  pb=$(stat -c %s "$tmp/prose.txt")
  cb=$(stat -c %s "$tmp/code.txt")
  want=$((CHUNKS * CTX - need + CTX / 4)) # the code part: the chunks left, and a quarter chunk over, so the last chunk is full
  [ "$pt" -ge "$need" ] || die "the prose source has $pt tokens, $need are needed for $half chunk(s) of $CTX" 65
  [ "$ct" -ge "$want" ] || die "the code source has $ct tokens, $want are needed" 65
  head -c $((need * pb / pt)) "$tmp/prose.txt" > "$out.tmp"
  printf '\n\n' >> "$out.tmp"
  head -c $((want * cb / ct)) "$tmp/code.txt" >> "$out.tmp"
  total=$(tok_count "$in" "$out.tmp")
  [ "$total" -ge $((CHUNKS * CTX + 1)) ] || { rm -f "$out.tmp"; rm -rf "$tmp"; die "the calibration text has $total tokens, $((CHUNKS * CTX + 1)) are needed for $CHUNKS chunk(s) of $CTX" 65; }
  mv -f "$out.tmp" "$out"
  say "calibration text $out: $total tokens (prose ~$need of $pt, code ~$want of $ct), $(stat -c %s "$out") bytes, md5 $(md5sum < "$out" | cut -d' ' -f1)"
  rm -rf "$tmp"
}

# ---- the rules against the table (Mac-checkable) ----
do_rules() {
  local f
  f=$(mktemp "${TMPDIR:-/tmp}/k31rules.XXXXXX") || die "no temporary file" 70
  make_rules "$f"
  if [ "$MAP" = ud ]; then
    echo "# rules for --map ud --type $TYPE (the file llama-quantize reads has no comment lines); bump layers gate/up [$BUMP_GU] down [$BUMP_DN]"
  else
    echo "# rules for --map $MAP --type $TYPE (the file llama-quantize reads has no comment lines); global type $QTYPE, token_embd $TOKEMB, output q6_k by flag"
  fi
  cat "$f"
  rm -f "$f"
}
do_check() {
  local f
  [ -f "$REF" ] || die "no reference table at $REF (--ref-table)" 66
  f=$(mktemp "${TMPDIR:-/tmp}/k31rules.XXXXXX") || die "no temporary file" 70
  make_rules "$f"
  if [ "$MAP" = ud ]; then
    [ "$TYPE" = IQ1_S ] || say "--type $TYPE has no reference table of its own: the check below holds the rules to $REF's IQ1_S types and will name the mismatches"
    splice check-rules --rules "$f" --table "$REF" || { rm -f "$f"; exit 65; }
  else
    say "--map $MAP: the rules name the routed experts and three small exceptions; every other tensor takes the global type $QTYPE. The types it gives are llama-quantize's own, so the proof is its plan (the plan step, on the box); here: each rule matches tensors of the K3 table"
    splice check-rules --rules "$f" --table "$REF" --partial || { rm -f "$f"; exit 65; }
  fi
  rm -f "$f"
}

# ---- the steps ----
BLIND=0 KEEP_LEASE=0 IMATRIX='' QFIRST='' IN_BYTES=0 IN_SHARDS=1 CARD='' PLAN_TSV=''
NICE=(nice -n 10 ionice -c2 -n7)
present_or_blind() { # a file a --dry-run of the smoke chain may not have yet (an earlier step writes it)
  [ -f "$1" ] || { [ "$DRY" = 1 ] && [ "$SMOKE" = 1 ]; }
}
planned_or_present() { # <file> <what>: a --dry-run of a chain shows the splice before its two inputs exist (the quantize step and prepare write them)
  [ -f "$1" ] && return 0
  [ "$DRY" = 1 ] || return 1
  [ "$SMOKE" = 1 ] || say "[dry-run] $2 $1 does not exist yet: a real run of this step is refused (66) until it does"
  return 0
}
need_in() { # IN: the K3-shaped set. A --dry-run may name a file an earlier step of the chain has not written yet (BLIND).
  BLIND=0
  if [ -z "$IN" ]; then
    [ "$DRY" = 1 ] || [ "$STEP" = plan ] || die "--in FILE (the K3-shaped set's first shard) or K3Q_IN" 64
    load_profile
    IN=$MODEL
    say "$([ "$DRY" = 1 ] && echo '[dry-run] ')no --in: the plan is shown over the K3 GGUF $IN (its headers are read, nothing else)"
  fi
  case $IN in /*) ;; *) IN=$PWD/$IN ;; esac
  if [ ! -f "$IN" ]; then
    present_or_blind "$IN" || die "no input at $IN" 66
    BLIND=1
    IN_BYTES=$(bytes_assumed)
    IN_SHARDS=$(shard_count "$IN")
    say "[dry-run] $IN is written by an earlier step of this chain"
    return 0
  fi
  IN_BYTES=$(set_bytes "$IN") || exit $? # a die inside $(...) ends only the subshell: a missing shard stops the runner here
  IN_SHARDS=$(shard_count "$IN")
}
plan_line() { # <step> <bytes> <what>
  local p b
  p=$(pred_of "$1" "$2")
  b=$(bound_of "$p")
  say "plan $1: $3; derived wall $(hms "$p"), bound ${b}s, card docs/cards/${CARD}.card"
}
preflight_shape() { # the input's names and dims against the table: refused before the long load finds it
  [ "$BLIND" = 0 ] || return 0
  [ -f "$REF" ] || die "no reference table at $REF (--ref-table)" 66
  splice check-shape --in "$IN" --names "$REF" || exit 65
}

do_imatrix() {
  need_in
  CARD=k31-imatrix
  [ "$SMOKE" = 0 ] || CARD=k31-smoke
  [ -x "$IMAT" ] || [ "$DRY" = 1 ] || die "no llama-imatrix at $IMAT: just build-k3-quant-tree" 66
  local im marker key calib pred bound
  im=$WORK/kimi-k31-c${CTX}x${CHUNKS}.imatrix.gguf
  [ "$SMOKE" = 0 ] || im=$WORK/smoke.imatrix.gguf
  IMATRIX=$im
  marker=$WORK/imatrix-$CTX-$CHUNKS.done
  [ "$SMOKE" = 0 ] || marker=$WORK/imatrix-smoke.done
  THREADS=${THREADS:-$(nproc)}
  pred=$(pred_of imatrix "$IN_BYTES")
  bound=$(bound_of "$pred")
  say "imatrix: input $IN ($IN_SHARDS shard(s), $IN_BYTES bytes), ctx $CTX, chunks $CHUNKS, threads $THREADS"
  plan_line imatrix "$IN_BYTES" "populate $K3_POPULATE_S s + $CHUNKS x $K3_CHUNK_S s (measured on the $K3_BYTES-byte K3 set) x $IN_BYTES/$K3_BYTES"
  if [ "$IN_BYTES" != "$K31_BYTES" ] && [ "$SMOKE" = 0 ]; then
    say "plan imatrix: for the 1.56 TB K3.1 file: $(hms "$(pred_of imatrix "$K31_BYTES")")"
  fi
  preflight_shape
  force_clean "$marker" "$im"
  partial_clean "$im.partial"
  if [ "$DRY" = 0 ]; then mkdir -p "$WORK"; fi
  calib=${CORPUS:-$WORK/calib-c${CTX}x${CHUNKS}.txt}
  if [ "$DRY" = 1 ]; then
    if [ -n "$CORPUS" ]; then
      say "[dry-run] calibration text: $CORPUS"
    else
      say "[dry-run] would build the calibration text $calib: $(((CHUNKS + 1) / 2)) chunk(s) of the engine tree's prose, then code (IK=${IK:-unset until the profile loads})"
    fi
  elif [ -n "$CORPUS" ]; then
    [ -f "$CORPUS" ] || die "no calibration text at $CORPUS" 66
  else
    build_calib "$calib" "$IN"
  fi
  if [ "$DRY" = 0 ]; then
    key=$(key_of imatrix "$CTX" "$CHUNKS" "$THREADS" "$(sha_of "$IMAT")" "$(set_ident "$IN")" "$(sha_of "$calib")")
    if marker_state "$marker" "$key" "$im"; then
      say "imatrix: done before ($marker, $im): skipped"
      return 0
    fi
  fi
  lease_on "$CARD"
  local t0=$SECONDS
  run imatrix "$bound" "${NICE[@]}" env CUDA_VISIBLE_DEVICES= "$IMAT" -m "$IN" -f "$calib" -o "$im.partial" --output-format gguf \
    -c "$CTX" -b "$CTX" -ub "$CTX" --chunks "$CHUNKS" --no-escape --no-ppl -t "$THREADS" || fail_step imatrix $?
  if [ "$DRY" = 1 ]; then lease_off; return 0; fi
  lease_off
  [ -s "$im.partial" ] || die "llama-imatrix ended 0 and wrote no $im.partial: see $LOGDIR/imatrix.log" 65
  splice list --in "$im.partial" --tsv --gguf-py off > "$WORK/imatrix.tsv" || die "$im.partial is not a GGUF file the reader takes (the imatrix output)" 65
  splice list --in "$IN" --tsv --gguf-py off > "$WORK/imatrix-in.tsv" || die "cannot read the input's tensor list" 65
  # Every routed-expert tensor of the input needs an entry (an IQ1_S tensor with none is refused by the quantizer, hours later).
  local miss
  miss=$(LC_ALL=C comm -23 <(cut -f1 "$WORK/imatrix-in.tsv" | grep -E '\.ffn_(gate|up|down)_exps\.weight$' | LC_ALL=C sort -u) \
    <(cut -f1 "$WORK/imatrix.tsv" | sed -E 's/\.(in_sum2|counts)$//' | LC_ALL=C sort -u))
  [ -z "$miss" ] || die "the imatrix has no entry for $(echo "$miss" | wc -l | tr -d ' ') routed-expert tensor(s) of the input, the first $(echo "$miss" | head -1): see $WORK/imatrix.tsv" 65
  mv -f "$im.partial" "$im"
  write_marker "$marker" "$key" "$im" "entries=$(wc -l < "$WORK/imatrix.tsv" | tr -d ' ') ctx=$CTX chunks=$CHUNKS calib_md5=$(md5sum < "$calib" | cut -d' ' -f1) calib=$(basename "$calib")"
  say "imatrix: done: $im, $(stat -c %s "$im") bytes, $(wc -l < "$WORK/imatrix.tsv" | tr -d ' ') tensors (in_sum2 and counts), every routed-expert tensor of the input covered; wall $((SECONDS - t0)) s for $IN_BYTES B in (derived $pred s)"
}

quant_args() { # <rules file> [<imatrix>]: sets QARGS, the flags both the plan and the run take
  QARGS=(--allow-requantize --tensor-type-file "$1" --output-tensor-type q6_k --token-embedding-type "$TOKEMB")
  [ -z "${2:-}" ] || QARGS+=(--imatrix "$2")
  [ "$IN_SHARDS" = 1 ] || QARGS+=(--keep-split)
}
plan_for() { # <rules file>: llama-quantize's own --dry-run over the input's headers into $WORK/plan-$TAG.{log,tsv} (PLAN_TSV), and its classes
  local rules=$1 log tsv trunk
  log=$LOGDIR/plan-$TAG.log
  tsv=$WORK/plan-$TAG.tsv
  quant_args "$rules"
  if [ "$DRY" = 1 ]; then
    if [ "$BLIND" = 1 ] || [ ! -x "$Q" ] || [ ! -f "$IN" ]; then
      say "[dry-run] the plan (llama-quantize --dry-run over the input's headers) is not run here: no llama-quantize or no input file on this host"
      PLAN_TSV=''
      return 0
    fi
    log=$(mktemp "${TMPDIR:-/tmp}/k31plan.XXXXXX") || die "no temporary file" 70
    tsv=$(mktemp "${TMPDIR:-/tmp}/k31plan.XXXXXX") || die "no temporary file" 70
    CUDA_VISIBLE_DEVICES='' timeout 300 "$Q" --dry-run "${QARGS[@]}" "$IN" "$WORK/plan-out.gguf" "${QTYPE:-$TYPE}" 1 > "$log" 2>&1 ||
      { tail -n 5 "$log" >&2; rm -f "$log" "$tsv"; die "llama-quantize --dry-run failed (above)" 65; }
    splice plan --log "$log" --out "$tsv" > /dev/null || { rm -f "$log" "$tsv"; exit 65; }
    say "[dry-run] plan of the quantize step over $IN (llama-quantize --dry-run, no data read), bytes per class:"
    splice classes --tsv "$tsv" | sed 's/^/    /'
    rm -f "$log" "$tsv"
    PLAN_TSV=''
    return 0
  fi
  run "plan-$TAG" 600 env CUDA_VISIBLE_DEVICES= "$Q" --dry-run "${QARGS[@]}" "$IN" "$WORK/plan-out.gguf" "${QTYPE:-$TYPE}" 1 || fail_step "plan-$TAG" $?
  splice plan --log "$log" --out "$tsv" || exit 65
  say "plan: bytes per class ($tsv; the log is $log):"
  splice classes --tsv "$tsv" | sed 's/^/    /'
  trunk=$(splice classes --tsv "$tsv" --field trunk) || exit 65
  if [ "$MAX_TRUNK" != 0 ] && [ "$trunk" -gt $((MAX_TRUNK * 1000000000)) ]; then
    die "the plan's trunk is $trunk B, over --max-trunk-gb $MAX_TRUNK ($((MAX_TRUNK * 1000000000)) B): the map is not the tight one, or the input is not K3's shape (--max-trunk-gb 0 turns the check off)" 65
  fi
  PLAN_TSV=$tsv
}

do_plan() {
  need_in
  CARD=k31-quantize
  [ "$DRY" = 1 ] || mkdir -p "$WORK"
  local rules
  rules=$(mktemp "${TMPDIR:-/tmp}/k31rules.XXXXXX") || die "no temporary file" 70
  make_rules "$rules"
  say "plan: --map $MAP, global type ${QTYPE:-$TYPE}, experts $TYPE, input $IN ($IN_SHARDS shard(s))"
  plan_for "$rules"
  rm -f "$rules"
}

do_quantize() {
  need_in
  CARD=k31-quantize
  [ "$SMOKE" = 0 ] || CARD=k31-smoke
  [ -x "$Q" ] || [ "$DRY" = 1 ] || die "no llama-quantize at $Q: just build-k3-quant-tree" 66
  local im qprefix qdir rules marker key pred bound qthreads pfirst args t0
  im=${IMATRIX:-$WORK/kimi-k31-c${CTX}x${CHUNKS}.imatrix.gguf}
  [ "$SMOKE" = 0 ] || im=${IMATRIX:-$WORK/smoke.imatrix.gguf}
  if [ ! -f "$im" ]; then
    [ "$DRY" = 1 ] || die "no imatrix at $im: run the imatrix step first (IQ1_S and IQ2_XXS are refused by the quantizer without one)" 66
    say "[dry-run] the imatrix $im is written by the imatrix step"
  fi
  qprefix=Kimi-K3.1-Pro-$TAG-k3shape
  [ "$SMOKE" = 0 ] || qprefix=k3shape
  qdir=$OUT_BASE/$qprefix
  rules=$WORK/rules-$TAG.txt
  marker=$WORK/quantize-$TAG.done
  THREADS=${THREADS:-$(nproc)}
  qthreads=${QTHREADS:-$THREADS}
  pred=$(pred_of quantize "$IN_BYTES")
  bound=$(bound_of "$pred")
  say "quantize: $IN ($IN_SHARDS shard(s), $IN_BYTES bytes) -> $qdir, --map $MAP, global type ${QTYPE:-$TYPE}, experts $TYPE, threads $qthreads"
  plan_line quantize "$IN_BYTES" "expert weights $K31_EXPERT_W x bytes/$K31_BYTES / ($QUANT_EFF_THREADS effective threads x $QUANT_RATE_MW Mweights/s, measured on 1 thread) + read $NVME_READ_GBS GB/s + write $OUT_PER_IN x bytes at $NVME_WRITE_GBS GB/s"
  if [ "$IN_BYTES" != "$K31_BYTES" ] && [ "$SMOKE" = 0 ]; then
    say "plan quantize: for the 1.56 TB K3.1 file: $(hms "$(pred_of quantize "$K31_BYTES")"); the rate band 3-6 Mweights/s/thread gives $(hms "$(QUANT_RATE_MW=6 pred_of quantize "$K31_BYTES")") to $(hms "$(QUANT_RATE_MW=3 pred_of quantize "$K31_BYTES")")"
  fi
  derive_bumps
  if [ "$MAP" = ud ]; then
    say "type map: ud, rules for $TYPE, IQ2_XXS bumps gate/up [${BUMP_GU:-none}] down [${BUMP_DN:-none}] (rules file $rules)"
  else
    say "type map: $MAP, the routed experts $TYPE and three small exceptions by rule, everything else the global type ${QTYPE} (rules file $rules)"
  fi
  local tmp_rules=$rules
  if [ "$DRY" = 1 ]; then
    tmp_rules=$(mktemp "${TMPDIR:-/tmp}/k31rules.XXXXXX") || die "no temporary file" 70
    make_rules "$tmp_rules"
    sed 's/^/      rule: /' "$tmp_rules"
    if [ -f "$REF" ]; then
      if [ "$MAP" = ud ]; then splice check-rules --rules "$tmp_rules" --table "$REF" || say "[dry-run] check-rules against $REF refused (above)"
      else splice check-rules --rules "$tmp_rules" --table "$REF" --partial || say "[dry-run] check-rules --partial against $REF refused (above)"; fi
    fi
    plan_for "$tmp_rules"
    rm -f "$tmp_rules"
  else
    mkdir -p "$WORK"
    make_rules "$rules"
    # The rules reproduce the table's types (ud) or name only what they must (the others), and the input's untouched tensors already
    # have the table's types: refused before the long run.
    if [ "$MAP" = ud ] && [ "$TYPE" = IQ1_S ]; then splice check-rules --rules "$rules" --table "$REF" --in "$IN" || exit 65
    else splice check-rules --rules "$rules" --table "$REF" --partial --in "$IN" || exit 65; fi
    plan_for "$rules"
  fi
  force_clean "$marker" "$qdir"
  partial_clean "$qdir.partial"
  if [ "$DRY" = 0 ]; then
    if [ -e "$qdir" ] && [ ! -f "$marker" ]; then die "$qdir exists and no done marker says this run made it: --force removes it, or --out-base elsewhere" 65; fi
    key=$(key_of quantize "$MAP" "$TYPE" "$(sha_of "$Q")" "$(set_ident "$IN")" "$(sha_of "$im")" "$(sha_of "$rules")" "$qthreads")
    if marker_state "$marker" "$key" "$qdir"; then
      QFIRST=$(named_first "$IN" "$qdir" "$qprefix")
      say "quantize: done before ($marker, $qdir): skipped"
      return 0
    fi
    mkdir -p "$qdir.partial"
  fi
  quant_args "$rules" "$im"
  args=("${QARGS[@]}")
  lease_on "$CARD"
  local tick=''
  if [ "$DRY" = 0 ] && [ "$SMOKE" = 0 ]; then
    # A progress line every two minutes (tensors converted so far): the rate is readable while the quantizer runs. fds 8 and 9 are
    # closed for it: a leftover ticker would hold the run lock and the lease.
    (while :; do sleep 120; printf '%s converted=%s\n' "$(date +%s)" "$(grep -c 'converting to' "$LOGDIR/quantize.log" 2> /dev/null)" >> "$WORK/quantize.progress"; done) 8>&- 9>&- &
    tick=$!
  fi
  t0=$SECONDS
  run quantize "$bound" "${NICE[@]}" env CUDA_VISIBLE_DEVICES= "$Q" "${args[@]}" "$IN" "$qdir.partial/$qprefix.gguf" "${QTYPE:-$TYPE}" "$qthreads"
  local rc=$?
  [ -z "$tick" ] || { kill "$tick" 2> /dev/null; wait "$tick" 2> /dev/null; }
  [ "$rc" = 0 ] || fail_step quantize "$rc"
  [ "$DRY" = 0 ] || { lease_off; QFIRST=$(named_first "$IN" "$qdir" "$qprefix"); return 0; }
  lease_off
  pfirst=$(named_first "$IN" "$qdir.partial" "$qprefix")
  [ -f "$pfirst" ] || die "llama-quantize ended 0 but $pfirst is not there (it names the shards from the output's prefix): $(find "$qdir.partial" -maxdepth 1 -type f | head -3 | tr '\n' ' ')" 65
  splice list --in "$pfirst" --tsv > "$WORK/quantize-$TAG.tsv" || die "the quantized set does not read back: $pfirst" 65
  # The output is the plan, tensor by tensor (names, dims, types); the ud map is held to the K3 table too.
  splice check-shape --in "$pfirst" --names "$PLAN_TSV" --types || die "the quantized set is not llama-quantize's own plan (names, dims and types, above)" 65
  if [ "$MAP" = ud ] && [ "$TYPE" = IQ1_S ]; then
    splice check-shape --in "$pfirst" --names "$REF" --types || die "the quantized set is not the reference table's names, dims and types (above)" 65
  fi
  mv "$qdir.partial" "$qdir"
  QFIRST=$(named_first "$IN" "$qdir" "$qprefix")
  write_marker "$marker" "$key" "$qdir" "tensors=$(wc -l < "$WORK/quantize-$TAG.tsv" | tr -d ' ') map=$MAP type=$TYPE"
  say "quantize: done: $qdir, $(du -sb "$qdir" | cut -f1) bytes; wall $((SECONDS - t0)) s for $IN_BYTES B in (derived $pred s); $(grep -E 'quantize time|total time|model size|quant size' "$LOGDIR/quantize.log" | tr -s ' ' | tr '\n' ';')"
  say "type table (the output's real tensors):"
  splice classes --in "$QFIRST" | sed 's/^/    /'
}

dsa_kv_args() { # <config.json> <arch>: one --kv value per line
  python3 -I - "$1" "$2" << 'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
d = (cfg.get("text_config") or cfg).get("dsa_config")
if not isinstance(d, dict):
    sys.exit("no text_config.dsa_config in " + sys.argv[1])
a = sys.argv[2]
for src, dst in (("index_topk", "top_k"), ("index_head_dim", "key_length"), ("index_num_heads", "head_count")):
    if not isinstance(d.get(src), int):
        sys.exit(f"dsa_config.{src} is missing or not an integer")
    print(f"{a}.attention.indexer.{dst}=u32:{d[src]}")
layers = d.get("dsa_layers")
if not (isinstance(layers, list) and layers and all(isinstance(x, int) for x in layers)):
    sys.exit("dsa_config.dsa_layers is missing, empty or not a list of integers")
# config.json counts layers from 1 (K3.1's dsa_layers 4, 8, ..., 92, 93 are blocks 3, 7, ..., 91, 92); the GGUF's blk.N counts from 0.
if min(layers) < 1:
    sys.exit("dsa_config.dsa_layers holds a layer below 1: the config counts layers from 1")
print(f"{a}.attention.indexer.layers=u32[]:" + ",".join(str(x - 1) for x in sorted(layers)))
PY
}
check_dsa_layers() { # <kv args...>: the layers key against the blocks the indexer file holds tensors for (a config's 1-based list is refused here)
  local want have
  want=$(printf '%s\n' "$@" | grep -o 'indexer\.layers=u32\[\]:[0-9,]*' | sed 's/.*://')
  [ -n "$want" ] || return 0
  have=$(splice list --in "$INDEXER" --tsv --gguf-py off | awk -F'\t' '$1 ~ /^blk\.[0-9]+\.indexer\./ { split($1, a, "."); print a[2] }' | sort -un | paste -sd, -)
  [ "$want" = "$have" ] ||
    die "the config's dsa_layers (as blocks: $want) are not the blocks the indexer file $INDEXER holds tensors for ($have): the config counts layers from 1, the GGUF from 0" 65
}

render_card() { # <README.md to write> <the set's first shard (in the .partial dir)> <the set's name>: tools/ref/k31-model-card.md.in, filled
  local out=$1 first=$2 name=$3 d rules im_marker note md5 corpus srcrev layers nlayers toolsline bcommit lcommit total flags
  d=$(mktemp -d "${TMPDIR:-/tmp}/k31card.XXXXXX") || die "no temporary directory" 70
  rules=$WORK/rules-$TAG.txt
  [ -f "$rules" ] || { rm -rf "$d"; die "no rules file $rules: the card states the type map; run the quantize step with this --map and --type first" 66; }
  im_marker=$WORK/imatrix-$CTX-$CHUNKS.done
  [ "$SMOKE" = 0 ] || im_marker=$WORK/imatrix-smoke.done
  [ -f "$im_marker" ] || { rm -rf "$d"; die "no imatrix marker $im_marker: the card states the corpus and the chunk count; run the imatrix step with this --ctx and --chunks first" 66; }
  note=$(sed -n '4p' "$im_marker")
  md5=$(printf '%s' "$note" | sed -n 's/.*calib_md5=\([0-9a-f]*\).*/\1/p')
  [ -n "$md5" ] || { rm -rf "$d"; die "$im_marker records no calibration text (calib_md5): run the imatrix step again (--force)" 65; }
  if printf '%s' "$note" | grep -q 'calib=calib-c'; then
    corpus="the engine tree's docs (*.md) and README, then its C++ sources, cut at the token count by the K3 tokenizer (md5 of the text: $md5)."
  else
    corpus="a text given with --corpus (md5 of the text: $md5)."
  fi
  srcrev=$SRC_REV
  [ -n "$srcrev" ] || srcrev=$(head -n 1 "$HF/.cache/huggingface/download/config.json.metadata" 2> /dev/null || true)
  case $srcrev in '' | *[!A-Za-z0-9._-]*) rm -rf "$d"; die "no source revision for the card: --source-rev REV (the HF commit the folder holds; the folder's download metadata at $HF/.cache/huggingface/download/config.json.metadata gave none)" 66 ;; esac
  layers=$(splice list --in "$INDEXER" --tsv --gguf-py off | awk -F'\t' '$1 ~ /^blk\.[0-9]+\.indexer\./ { split($1, a, "."); print a[2] }' | sort -un | paste -sd, -)
  nlayers=$(printf '%s\n' "$layers" | tr ',' '\n' | grep -c .)
  [ "$nlayers" -gt 0 ] || { rm -rf "$d"; die "the indexer file $INDEXER holds no blk.N.indexer.* tensor: the card names the DSA layers" 65; }
  lcommit=$(git -c "safe.directory=$TREE" -C "$TREE" rev-parse --short=9 HEAD 2> /dev/null) || lcommit=unknown # -c: the tree is another user's, as root reads it
  if [ -n "${K3Q_TOOLS_COMMIT:-}" ]; then
    bcommit=$K3Q_TOOLS_COMMIT
  else
    bcommit=$(git -C "$ROOT" rev-parse --short=9 HEAD 2> /dev/null) || { bcommit=unknown; say "the bloomery tree at $ROOT is not a git checkout (the box's copy has no .git): the card says the commit is unknown; K3Q_TOOLS_COMMIT=<sha> names it"; }
  fi
  toolsline="llama.cpp \`$lcommit\` (mainline; llama-quantize sha256 \`$(sha_of "$Q" | cut -c1-12)\`, llama-imatrix sha256 \`$(sha_of "$IMAT" | cut -c1-12)\`); bloomery \`$bcommit\` (tools/ref/gguf-splice.py, tools/ref/k31-quant.sh)."
  flags="llama-quantize --allow-requantize --imatrix <imatrix.gguf> --tensor-type-file <rules.txt> --output-tensor-type q6_k --token-embedding-type $TOKEMB --keep-split <k3-shaped-input.gguf> <output.gguf> ${QTYPE:-$TYPE} <threads>"
  splice classes --in "$first" --md > "$d/classes.md" || { rm -rf "$d"; exit 65; }
  splice list --in "$first" --gguf-py off | awk '/^shard / { n = split($3, p, "/"); t = $4; b = $6; sub(/^tensors=/, "", t); sub(/^bytes=/, "", b); rows = rows "| " $2 " | " p[n] " | " t " | " b " |\n" }
    END { printf "| shard | file | tensors | bytes |\n|---|---|---|---|\n%s", rows }' > "$d/shards.md"
  total=$(splice list --in "$first" --gguf-py off | awk '/^shard / { b = $6; sub(/^bytes=/, "", b); s += b } END { printf "%.2f", s / 1e9 }')
  splice fill --template "$ROOT/tools/ref/k31-model-card.md.in" --out "$out" \
    --set "NAME=$name" --set "ARCH=$ARCH" --set "MAP=$MAP" --set "QTYPE=${QTYPE:-$TYPE}" --set "EXPERT_TYPE=$TYPE" --set "TOTAL_GB=$total" \
    --set "INDEXER_COUNT=$nlayers" --set "INDEXER_LAYERS=$layers" --set "IMATRIX_CHUNKS=$CHUNKS" --set "IMATRIX_CTX=$CTX" \
    --set "IMATRIX_TOKENS=$((CHUNKS * CTX))" --set "IMATRIX_CORPUS=$corpus" --set "SOURCE_REV=$srcrev" --set "TOOLS=$toolsline" \
    --set "QUANT_FLAGS=$flags" --file "CLASS_TABLE=$d/classes.md" --file "RULES=$rules" --file "SHARD_TABLE=$d/shards.md" || { rm -rf "$d"; exit 65; }
  rm -rf "$d"
}

do_splice() {
  CARD=k31-splice
  [ "$SMOKE" = 0 ] || CARD=k31-smoke
  [ -n "$INDEXER" ] || die "--indexer FILE (the indexer tensors' GGUF, prepare's --only output) or K3Q_INDEXER" 64
  planned_or_present "$INDEXER" "the indexer file" || die "no indexer file at $INDEXER" 66
  local sin sprefix sdir pfirst kvargs=() kv marker key pred bound lic t0
  lic=${LICENSE_F:-$HF/LICENSE}
  [ "$DRY" = 1 ] || [ -f "$lic" ] || die "no LICENSE at $lic (--license FILE or --hf-folder): the output set carries the source repository's" 66
  sin=${QFIRST:-$IN}
  if [ -z "$sin" ]; then
    # the default: the set the quantize step of this --type wrote
    local d=$OUT_BASE/Kimi-K3.1-Pro-$TAG-k3shape
    sin=$(first_match "$d"/*-00001-of-*.gguf "$d"/*.gguf)
    if [ -z "$sin" ] && [ "$DRY" = 1 ]; then # the plan: the name the quantize step will give shard 1, with the K3 set's shard count
      load_profile
      sin=$(named_first "$MODEL" "$d" "Kimi-K3.1-Pro-$TAG-k3shape")
    fi
    [ -n "$sin" ] || die "--in FILE: the quantized K3-shaped set's first shard (nothing at $d)" 64
  fi
  planned_or_present "$sin" "the quantized set" || die "no quantized set at $sin" 66
  sprefix=Kimi-K3.1-Pro-$TAG
  [ "$SMOKE" = 0 ] || sprefix=final
  sdir=$OUT_BASE/$sprefix
  pfirst=$(named_first "$sin" "$sdir.partial" "$sprefix")
  marker=$WORK/splice-$TAG.done
  pred=$(pred_splice)
  bound=$(bound_of "$pred")
  if [ -n "$DSA" ] && [ ! -f "$DSA" ] && [ "$DRY" = 1 ] && [ "$SMOKE" = 1 ]; then
    say "[dry-run] $DSA is written by the smoke (4 words of --kv from it: top_k, key_length, head_count, layers)"
  elif [ -n "$DSA" ]; then
    [ -f "$DSA" ] || die "no config at $DSA" 66
    while IFS= read -r kv; do kvargs+=(--kv "$kv"); done < <(dsa_kv_args "$DSA" "$ARCH") || true
    [ "${#kvargs[@]}" -gt 0 ] || die "$DSA: no dsa_config keys came out (above)" 65
  else
    say "no --dsa-config: the file gets no $ARCH.attention.indexer.* keys (the reader needs them: pass the K3.1 config.json)"
  fi
  if [ -f "$INDEXER" ]; then check_dsa_layers ${kvargs[@]+"${kvargs[@]}"}; elif [ "${#kvargs[@]}" -gt 0 ]; then say "[dry-run] the indexer file is not there: the layers key cannot be checked against its blocks yet"; fi
  say "splice: $sin + $INDEXER -> $sdir, architecture $ARCH, ${#kvargs[@]} word(s) of --kv"
  if [ "$SMOKE" = 1 ]; then
    say "plan splice: the smoke's shards (a few MB each; the 1.56 TB projection does not apply), bound ${bound}s, card docs/cards/${CARD}.card"
  else
    say "plan splice: the receiving shard read, written and read back ($SPLICE_SHARD_BYTES B at $NVME_READ_GBS/$NVME_WRITE_GBS GB/s), the first shard rewritten, the others hard links: $(hms "$pred"), bound ${bound}s, card docs/cards/${CARD}.card"
  fi
  if [ "$DRY" = 0 ]; then
    [ "$(stat -c %d "$(dirname "$sin")")" = "$(stat -c %d "$OUT_BASE")" ] ||
      die "$sin and $OUT_BASE are on different filesystems: --link would copy the whole set (594 GB); give --out-base on the set's filesystem" 64
    force_clean "$marker" "$sdir"
    partial_clean "$sdir.partial"
    [ ! -e "$sdir" ] || [ -f "$marker" ] || die "$sdir exists and no done marker says this run made it: --force removes it" 65
    key=$(key_of splice "$ARCH" "$(sha_of "$SPLICE")" "$(set_ident "$sin")" "$(stat -c '%n %s %Y' "$INDEXER")" ${kvargs[@]+"${kvargs[@]}"})
    if marker_state "$marker" "$key" "$sdir"; then
      say "splice: done before ($marker, $sdir): skipped"
      return 0
    fi
    mkdir -p "$sdir.partial"
  fi
  lease_on "$CARD"
  t0=$SECONDS
  run splice "$bound" python3 -I "$SPLICE" add --in "$sin" --from "$INDEXER" --take 'blk.*.indexer.*' --out "$pfirst" \
    --arch "$ARCH" --link ${kvargs[@]+"${kvargs[@]}"} || fail_step splice $?
  lease_off
  [ "$DRY" = 0 ] || return 0
  splice list --in "$pfirst" --tsv > "$WORK/splice-$TAG.tsv" || die "the spliced set does not read back: $pfirst" 65
  local nidx ntot nq nint
  nidx=$(grep -c '\.indexer\.' "$WORK/splice-$TAG.tsv")
  ntot=$(wc -l < "$WORK/splice-$TAG.tsv" | tr -d ' ')
  nint=$(splice list --in "$INDEXER" --tsv --gguf-py off | grep -c '\.indexer\.')
  nq=$(splice list --in "$sin" --tsv --gguf-py off | wc -l | tr -d ' ')
  [ "$nidx" = "$nint" ] && [ "$ntot" = $((nq + nint)) ] ||
    die "the spliced set has $ntot tensors ($nidx indexer ones); expected $nq + $nint = $((nq + nint)) (the indexer file's $nint)" 65
  cp -- "$lic" "$sdir.partial/LICENSE" || die "cannot copy $lic into $sdir.partial" 70
  render_card "$sdir.partial/README.md" "$pfirst" "$sprefix"
  mv "$sdir.partial" "$sdir"
  write_marker "$marker" "$key" "$sdir" "tensors=$ntot indexer=$nidx"
  say "splice: done: $sdir, $ntot tensors ($nidx indexer), LICENSE and README.md written; wall $((SECONDS - t0)) s; $(splice list --in "$(named_first "$sin" "$sdir" "$sprefix")" --gguf-py off | grep -m1 '^architecture=')"
}

# ---- the smoke: the chain on the K3 fixture, one lease, one card ----
do_upload() {
  CARD=none
  local dir=${IN:-$OUT_BASE/Kimi-K3.1-Pro-$TAG} marker key up_marker uvx bound t0 who
  marker=$WORK/splice-$TAG.done
  up_marker=$WORK/upload-$TAG.done
  bound=${K3Q_UPLOAD_BOUND_S:-86400}
  case $bound in '' | *[!0-9]* | 0) die "K3Q_UPLOAD_BOUND_S wants whole seconds above 0, got '$bound'" ;; esac
  say "upload: $dir -> https://huggingface.co/$REPO (model repo), bound ${bound}s"
  # Refusals by name, before anything is sent: the directory is the splice step's finished output, with its card and its license.
  if [ ! -d "$dir" ]; then
    [ "$DRY" = 1 ] || die "no directory $dir (--in DIR: the splice step's output)" 66
    say "[dry-run] $dir does not exist yet"
  fi
  if [ "$DRY" = 0 ]; then
    [ -f "$marker" ] || die "no .done marker of the splice step ($marker): $dir is not known to be a finished splice output (run the splice step with this --map and --type)" 65
    [ "$(sed -n 's/^out=//p' "$marker")" = "$dir" ] || die "$marker says the splice step wrote $(sed -n 's/^out=//p' "$marker"), not $dir" 65
    [ -f "$dir/README.md" ] || die "no README.md in $dir: the splice step writes the model card" 65
    [ -f "$dir/LICENSE" ] || die "no LICENSE in $dir: the splice step copies the source's" 65
    head -n 12 "$dir/README.md" | grep -q '^license: other' || die "$dir/README.md has no front matter with 'license: other'" 65
    ! grep -q '@[A-Z][A-Z0-9_]*@' "$dir/README.md" || die "$dir/README.md still has an unfilled @PLACEHOLDER@" 65
    uvx=$(command -v uvx) || die "no uvx on PATH (/usr/local/bin/uvx on the box)" 66
    who=$("$uvx" --from huggingface_hub hf auth whoami 2>&1 | head -n 1) || true
    case $who in '' | *[Nn]ot\ logged* | *rror*) die "not logged in to the Hub (hf auth whoami: '$who'): log in as the user, never paste a token here" 66 ;; esac
    say "upload: logged in as $who"
    key=$(key_of upload "$REPO" "$dir" "$(sed -n 's/^key=//p' "$marker")")
    if marker_state "$up_marker" "$key" "$dir"; then
      say "upload: done before ($up_marker): skipped (--force sends again; the Hub skips what it has)"
      return 0
    fi
    force_clean "$up_marker"
  else
    uvx=uvx
  fi
  t0=$SECONDS
  run upload "$bound" "${NICE[@]}" "$uvx" --from huggingface_hub hf upload-large-folder "$REPO" "$dir" --repo-type model || fail_step upload $?
  [ "$DRY" = 1 ] && return 0
  write_marker "$up_marker" "$key" "$dir" "repo=$REPO"
  say "upload: done: $REPO <- $dir; wall $((SECONDS - t0)) s"
}

do_prepare() {
  CARD=none
  local bin=${PREPARE_BIN:-$ROOT/target/release/prepare} out_a out_b usage f bound marker key a_first i_first extra=() t0
  out_a=$OUT_BASE/Kimi-K3.1-Pro-k3shape-src/Kimi-K3.1-Pro-k3shape
  out_b=$WORK/indexer-only/Kimi-K3.1-Pro-indexer
  marker=$WORK/prepare.done
  bound=${K3Q_PREPARE_BOUND_S:-43200}
  case $bound in '' | *[!0-9]* | 0) die "K3Q_PREPARE_BOUND_S wants whole seconds above 0, got '$bound'" ;; esac
  [ -z "$SHARD_GIB" ] || extra+=(--shard-gib "$SHARD_GIB")
  [ -z "$FOLLOW" ] || extra+=(--follow "$FOLLOW")
  say "prepare: $HF -> $out_a-NNNNN-of-NNNNN.gguf (K3-shaped: --drop 'blk.*.indexer.*' --arch kimi-k3) and $out_b-... (--only 'blk.*.indexer.*'), bound ${bound}s each"
  if [ "$DRY" = 0 ]; then
    [ -d "$HF" ] || die "no HF folder at $HF (--hf-folder)" 66
    [ -x "$bin" ] || die "no built prepare at $bin (--prepare-bin BIN): build it from the tree whose prepare has --drop, --only and --arch (round k31hf)" 66
    usage=$("$bin" 2>&1 < /dev/null || true)
    for f in --drop --only --arch; do
      case $usage in *"$f"*) ;; *) die "the built prepare at $bin has no $f in its usage: build it from a tree that has --drop, --only and --arch (round k31hf)" 66 ;; esac
    done
    key=$(key_of prepare "$(sha_of "$bin")" "$(stat -c '%s %Y' "$HF/config.json")" "$SHARD_GIB" "$FOLLOW")
    a_first=$(first_match "$out_a"-00001-of-*.gguf) || a_first=''
    i_first=$(first_match "$out_b"-00001-of-*.gguf) || i_first=''
    if [ -f "$marker" ] && [ "$(sed -n 's/^key=//p' "$marker")" = "$key" ] && [ -n "$a_first" ] && [ -n "$i_first" ]; then
      say "prepare: done before ($marker): $a_first and $i_first"
      return 0
    fi
    [ ! -f "$marker" ] || [ "$FORCE" = 1 ] || [ "$(sed -n 's/^key=//p' "$marker")" = "$key" ] || die "$marker records another prepare binary or folder than this run's: --force runs it again" 65
    if [ -n "$a_first" ] && [ "$FORCE" = 0 ] && [ ! -f "$marker" ]; then die "$a_first exists and no marker says this run made it: --force, or --out-base elsewhere" 65; fi
    mkdir -p "$(dirname "$out_a")" "$(dirname "$out_b")"
  fi
  t0=$SECONDS
  run prepare-k3shape "$bound" "${NICE[@]}" "$bin" "$HF" "$out_a" --drop 'blk.*.indexer.*' --arch kimi-k3 ${extra[@]+"${extra[@]}"} || fail_step prepare-k3shape $?
  run prepare-indexer "$bound" "${NICE[@]}" "$bin" "$HF" "$out_b" --only 'blk.*.indexer.*' || fail_step prepare-indexer $?
  [ "$DRY" = 0 ] || return 0
  a_first=$(first_match "$out_a"-00001-of-*.gguf) || die "prepare ended 0 and wrote no $out_a-00001-of-*.gguf (log $LOGDIR/prepare-k3shape.log)" 65
  i_first=$(first_match "$out_b"-00001-of-*.gguf "$out_b".gguf) || die "prepare ended 0 and wrote no indexer-only file at $out_b (log $LOGDIR/prepare-indexer.log)" 65
  # held to K3's shape before an imatrix run reads it: the names and dims of the table, no indexer tensor; the indexer file only indexer tensors
  splice check-shape --in "$a_first" --names "$REF" || die "prepare's K3-shaped output is not K3's names and dims (above)" 65
  [ "$(splice list --in "$i_first" --tsv --gguf-py off | grep -vc '\.indexer\.')" = 0 ] || die "the indexer-only output $i_first holds a tensor that is not an indexer one" 65
  [ "$(splice list --in "$i_first" --tsv --gguf-py off | grep -c '\.indexer\.')" -gt 0 ] || die "the indexer-only output $i_first holds no indexer tensor" 65
  write_marker "$marker" "$key" "$out_a" "in=$a_first indexer=$i_first"
  say "prepare: done: K3-shaped $a_first, indexer-only $i_first; wall $((SECONDS - t0)) s. Next: k31-quant.sh all (they are the defaults of --in and --indexer)"
}

use_prepared() { # --in and --indexer default to prepare's outputs when its marker is there
  local m=$WORK_BASE/prepare.done v
  if [ "$SMOKE" = 1 ] || [ ! -f "$m" ]; then return 0; fi
  if [ -z "$IN" ] && [ "$STEP" != splice ] && [ "$STEP" != upload ]; then
    v=$(sed -n 's/^in=\([^ ]*\).*/\1/p' "$m")
    [ -z "$v" ] || { IN=$v; say "--in: prepare's K3-shaped output $IN"; }
  fi
  if [ -z "$INDEXER" ]; then
    v=$(sed -n 's/.* indexer=\([^ ]*\).*/\1/p' "$m")
    [ -z "$v" ] || { INDEXER=$v; say "--indexer: prepare's indexer-only output $INDEXER"; }
  fi
}

make_indexer_script() { # <file>: a synthetic indexer GGUF through gguf-py's own writer (a third party to gguf-splice's)
  cat > "$1" << 'PY'
import sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1] + "/gguf-py")
import gguf
out, layers = sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
w = gguf.GGUFWriter(out, "kimi-k3")
rng = np.random.default_rng(7)
for l in layers:
    for name, shape in ((f"blk.{l}.indexer.attn_q_b.weight", (16, 24)), (f"blk.{l}.indexer.attn_k.weight", (8, 512)),
                        (f"blk.{l}.indexer.proj.weight", (4, 512)), (f"blk.{l}.indexer.k_norm.weight", (8,)), (f"blk.{l}.indexer.k_norm.bias", (8,))):
        bits = (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)  # bfloat16's bits
        w.add_tensor(name, bits, raw_dtype=gguf.GGMLQuantizationType.BF16)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out}: layers {layers}")
PY
}
smoke_rate() { # <the input's tensor list> <the quantized set's list>: the quantizer's rate on what it converted
  local ms w
  ms=$(sed -n 's/.*quantize time = *\([0-9.]*\) ms.*/\1/p' "$LOGDIR/quantize.log" | tail -1)
  [ -n "$ms" ] || die "no 'quantize time' line in $LOGDIR/quantize.log" 65
  # the weights of the tensors whose type changed, the routed experts' apart from the rest
  w=$(awk -F'\t' 'NR == FNR { ty[$1] = $2; next } ($1 in ty) && ty[$1] != $2 { n = split($3, d, "x"); p = 1; for (i = 1; i <= n; i++) p *= d[i]; if ($1 ~ /ffn_(down|gate|up)_exps/) { e += p; ec++ } else { o += p; oc++ } } END { printf "%d %d %d %d", e, ec, o, oc }' "$1" "$2")
  awk -v ms="$ms" -v w="$w" -v th="$QTHREADS" -v ref="$QUANT_RATE_MW" 'BEGIN {
      split(w, a, " ")
      r = (a[1] + a[3]) / (ms / 1000) / th / 1e6
      printf "smoke: the quantizer converted %d routed-expert tensors (%d weights) and %d other tensors (%d weights) in %.0f ms on %d thread(s): %.2f Mweights/s/thread over all of them", a[2], a[1], a[4], a[3], ms, th, r
      if (a[4] == 0) printf " = the experts alone (the figure includes the read and write of the fixture: a lower bound); the runner'"'"'s QUANT_RATE_MW is %s\n", ref
      else printf "; the other tensors are in that time, so this is NOT the experts'"'"' rate (the stand-in is not at the map'"'"'s trunk types)\n"
    }' | sed 's/^/k31-quant.sh: /'
}
do_smoke() {
  local fx=${K3Q_FIXTURE:-/models/fixtures/kimik3/kimi-k3-fixture-00001-of-00001.gguf} s=$WORK first_src first_like first_strip first_rt n=3 b
  CARD=k31-smoke
  if [ "$DRY" = 0 ]; then
    [ -f "$fx" ] || die "no fixture at $fx" 66
    for b in "$Q" "$IMAT" "$TOKENIZE" "$GSPLIT"; do [ -x "$b" ] || die "no $b: just build-k3-quant-tree" 66; done
    [ -n "${LCPP:-}" ] && [ -d "$LCPP/gguf-py/gguf" ] || die "no gguf-py under LCPP=${LCPP:-unset}: the smoke's indexer file is written by it" 66
    case $s in */smoke) rm -rf -- "$s" ;; *) die "the smoke directory '$s' does not end in /smoke: refusing to clear it" 70 ;; esac
    mkdir -p "$s/split" "$s/k31like" "$s/stripped" "$s/roundtrip" "$s/out"
  fi
  THREADS=${THREADS:-$(nproc)}
  QTHREADS=1
  REF=$s/ref.tsv
  KEEP_LEASE=1
  lease_on k31-smoke
  if [ "$USE_LEASE" = 1 ]; then
    say "smoke: fixture $fx -> $s under one CPU lease (card docs/cards/k31-smoke.card); every wall below is the smoke's, not a claim"
  else
    say "smoke: fixture $fx -> $s, no lease (--lease takes the CPU lease and makes the 1-thread rate a measurement); every wall below is the smoke's, not a claim"
  fi
  if [ "$DRY" = 0 ]; then
    splice list --in "$fx" --tsv --gguf-py off > "$REF" || die "cannot read the fixture's tensor list" 65
    # the stand-in's trunk is already the map's result (the same global type and flags): the smoke's quantize then converts the experts only,
    # as the real run's time is the experts' (the tight map's trunk is 2 % of the weights); the ud map keeps its Q6_K stand-in
    if [ "$MAP" = ud ]; then gen_rules mxfp4 '' '' '' > "$s/rules-src.txt"; else gen_rules_trunk mxfp4 > "$s/rules-src.txt"; fi
    make_indexer_script "$s/make-indexer.py"
    printf '{"text_config": {"dsa_config": {"dsa_layers": [2, 3], "index_topk": 2048, "index_head_dim": 8, "index_num_heads": 2}}}\n' > "$s/dsa.json"
    printf 'the smoke'"'"'s stand-in for the source LICENSE\n' > "$s/LICENSE"
  else
    # the plan reads the fixture's header for the reference table (the dry-run writes nothing: a temporary file, removed below)
    [ -f "$fx" ] || die "no fixture at $fx" 66
    REF=$(mktemp "${TMPDIR:-/tmp}/k31smoke-ref.XXXXXX") || die "no temporary file" 70
    splice list --in "$fx" --tsv --gguf-py off > "$REF" || { rm -f "$REF"; die "cannot read the fixture's tensor list ($fx)" 65; }
  fi
  # the K3.1 stand-in: the fixture's tensors, the experts as MXFP4 (llama-quantize's own writer), in a split set the way a --keep-split run reads it
  run smoke-src 600 env CUDA_VISIBLE_DEVICES= "$Q" --allow-requantize --tensor-type-file "$s/rules-src.txt" --output-tensor-type q6_k \
    --token-embedding-type "$TOKEMB" "$fx" "$s/src.gguf" "${QTYPE:-Q6_K}" "$THREADS" || fail_step smoke-src $?
  run smoke-split 300 "$GSPLIT" --split --split-max-tensors 40 "$s/src.gguf" "$s/split/src" || fail_step smoke-split $?
  first_src=$s/split/src-00001-of-0000$n.gguf
  if [ "$DRY" = 0 ]; then
    first_src=$(first_match "$s"/split/src-00001-of-*.gguf) || die "llama-gguf-split wrote no $s/split/src-00001-of-*.gguf" 65
    n=$(shard_count "$first_src")
  fi
  say "smoke: stand-in set $first_src ($n shard(s))"
  run smoke-indexer 300 python3 -I "$s/make-indexer.py" "${LCPP:-/home/user/llama.cpp-kimik3}" "$s/idx.gguf" 1,2 || fail_step smoke-indexer $?
  first_like=$(named_first "$first_src" "$s/k31like" k31like)
  first_strip=$(named_first "$first_src" "$s/stripped" stripped)
  first_rt=$(named_first "$first_src" "$s/roundtrip" rt)
  # strip: add the synthetic indexer tensors to the stand-in (what prepare's full output looks like), then strip them out and compare
  run smoke-add 300 python3 -I "$SPLICE" add --in "$first_src" --from "$s/idx.gguf" --take 'blk.*.indexer.*' --out "$first_like" --link || fail_step smoke-add $?
  run smoke-strip 300 python3 -I "$SPLICE" strip --in "$first_like" --out "$first_strip" --drop 'blk.*.indexer.*' --link || fail_step smoke-strip $?
  run smoke-compare 300 python3 -I "$SPLICE" compare --a "$first_src" --b "$first_strip" || fail_step smoke-compare $?
  [ "$DRY" = 1 ] || say "smoke: strip: $(tail -n 1 "$LOGDIR/smoke-compare.log")"
  # imatrix, quantize, splice: the real steps, on the stripped set
  IN=$first_strip
  do_imatrix
  do_quantize
  INDEXER=$s/idx.gguf
  DSA=$s/dsa.json
  LICENSE_F=$s/LICENSE
  SRC_REV=smoke-fixture
  do_splice
  # read back: strip the indexer out of the final set again and compare its tensors with the quantized set's
  local final
  final=$(named_first "$first_src" "$OUT_BASE/final" final)
  run smoke-roundtrip 300 python3 -I "$SPLICE" strip --in "$final" --out "$first_rt" --drop 'blk.*.indexer.*' || fail_step smoke-roundtrip $?
  run smoke-compare-q 300 python3 -I "$SPLICE" compare --a "$QFIRST" --b "$first_rt" --tensors-only || fail_step smoke-compare-q $?
  if [ "$DRY" = 0 ]; then
    say "smoke: roundtrip: $(tail -n 1 "$LOGDIR/smoke-compare-q.log")"
    [ -f "$OUT_BASE/final/LICENSE" ] || die "smoke: the splice wrote no LICENSE into $OUT_BASE/final" 65
    [ "$(head -n 1 "$OUT_BASE/final/README.md")" = '---' ] || die "smoke: $OUT_BASE/final/README.md does not open with front matter" 65
    grep -q '^license: other' "$OUT_BASE/final/README.md" || die "smoke: $OUT_BASE/final/README.md has no 'license: other'" 65
    grep -q '^base_model: moonshotai/Kimi-K3.1-Pro' "$OUT_BASE/final/README.md" || die "smoke: $OUT_BASE/final/README.md has no base_model" 65
    ! grep -q '@[A-Z][A-Z0-9_]*@' "$OUT_BASE/final/README.md" || die "smoke: $OUT_BASE/final/README.md has an unfilled placeholder" 65
    say "smoke: model card $OUT_BASE/final/README.md ($(wc -l < "$OUT_BASE/final/README.md" | tr -d ' ') lines), LICENSE copied; the card's table:"
    sed -n '/^| class/,/^| total/p' "$OUT_BASE/final/README.md" | sed 's/^/    /'
    splice list --in "$final" --gguf-py off | grep -E '^(architecture|shard)' | sed 's/^/k31-quant.sh: smoke: final: /'
    smoke_rate "$WORK/imatrix-in.tsv" "$WORK/quantize-smoke.tsv"
    say "smoke: PASS"
  fi
  KEEP_LEASE=0
  lease_off
  [ "$DRY" = 0 ] || rm -f "$REF"
}

# ---- main ----
if [ "$DETACH" = 1 ] && [ -z "${K3Q_DETACHED:-}" ]; then
  mkdir -p "$WORK_BASE"
  base=$WORK_BASE/$STEP
  rm -f "$base.rc"
  K3Q_DETACHED=1 K3Q_RC=$base.rc setsid nohup bash "$SELF" "${ORIG_ARGS[@]}" > "$base.runner.log" 2>&1 8>&- 9>&- < /dev/null &
  echo $! > "$base.runner.pid"
  say "detached: pid $(cat "$base.runner.pid"), log $base.runner.log, exit status in $base.rc when it ends (poll: kill -0 \$(cat $base.runner.pid))"
  exit 0
fi
# shellcheck disable=SC2064
[ -z "${K3Q_RC:-}" ] || trap 'echo $? > "$K3Q_RC"' EXIT
do_calib() {
  need_in
  [ "$BLIND" = 0 ] || die "calib: no input at $IN" 66
  [ "$DRY" = 0 ] || die "calib builds a file: it has no --dry-run (the imatrix step's --dry-run says where it goes)" 64
  mkdir -p "$WORK"
  build_calib "${CORPUS:-$WORK/calib-c${CTX}x${CHUNKS}.txt}" "$IN"
}

case $STEP in
  rules) do_rules ;;
  check) do_check ;;
  calib)
    load_profile
    do_calib ;;
  imatrix | quantize | splice | all | smoke | plan | prepare | upload)
    [ "$DRY" = 1 ] || mkdir -p "$WORK"
    load_profile
    load_lease
    case $STEP in imatrix | quantize | splice | smoke | prepare | upload | all) [ "$DRY" = 1 ] || acquire_lock ;; esac
    case $STEP in imatrix | quantize | splice | all | plan) use_prepared ;; esac
    if [ "$DRY" = 0 ]; then
      command -v nice > /dev/null 2>&1 || die "no nice on this host: imatrix, quantize, prepare and upload run under it" 66
      command -v ionice > /dev/null 2>&1 || die "no ionice on this host: imatrix, quantize, prepare and upload run under it" 66
    fi
    [ -x "$SPLICE" ] || [ -f "$SPLICE" ] || die "no $SPLICE" 66
    command -v python3 > /dev/null 2>&1 || die "no python3" 66
    case $STEP in imatrix | quantize | all) [ "$DRY" = 1 ] || [ -n "$IN" ] || die "--in FILE (the K3-shaped set's first shard) or K3Q_IN" 64 ;; esac
    case $STEP in
      imatrix) do_imatrix ;;
      quantize) do_quantize ;;
      splice) do_splice ;;
      all)
        [ -n "$INDEXER" ] || die "all needs --indexer FILE before it starts: the splice is the last step, and the first two take hours" 64
        do_imatrix
        do_quantize
        do_splice ;;
      smoke) do_smoke ;;
      plan) do_plan ;;
      prepare) do_prepare ;;
      upload) do_upload ;;
    esac ;;
esac
