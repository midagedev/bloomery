#!/usr/bin/env bash
# ik-ppl.sh — wikitext-2 perplexity of one ik_llama.cpp tree on the V4.1 file, CPU only, under the
# machine-wide CPU lease. It is the measurement our ik port's PR quotes (c2048, 4 chunks, CPU only), so
# two trees run back to back through it are an A/B of the port. It also writes ik's KL-divergence base
# file and scores a tree against one: the ik half of B5's paired perplexity comparison.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-ppl.sh <tree> <tag> [options]'
#   just ik-ppl <tree> <tag> [options]
#
#   <tree>       an ik tree with build/bin/llama-perplexity built in it, e.g. /home/user/ik-idxkey
#   <tag>        names the run: $BLOOMERY_DATA/ikppl/<tag>.log (everything) and <tag>.out (the tool's
#                stdout), and the result line carries it
#   --chunks N   chunks to score (default 4, the PR's); 1 is a smoke run
#   --ctx N      ids per chunk (default 2048, the PR's); ik scores the second half of each chunk
#   --batch N    logical batch, at most --ctx (default: --ctx)
#   --ubatch N   physical batch (-ub), at most --batch (default: ik's own, 512)
#   --kld-base   also write the base file $BLOOMERY_DATA/ikppl/<tag>.kld (--kl-divergence-base)
#   --kld BASE   score the tree against $BLOOMERY_DATA/ikppl/BASE.kld (--kl-divergence), BASE being the
#                base run's tag; --ctx and --chunks must be the base's
#
# The model defaults to the profile's MODEL (tools/ref/models/deepseek41.sh), shard 1 of the served file
# set: the file the oracle and the engine read. A second 300+ GB file set would evict it from the page
# cache of this 256 GiB machine, and every V4.1 run after this one would page it back in. MODEL=<gguf> in
# the box command overrides it; from the Mac side BLOOMERY_REF_MODEL does (box.sh carries it over). The
# PR's own number was measured on the plain Q3_K_M file set, /models/DeepSeek-V4.1-Flash-Q3_K_M/, not on
# this graft, so this runner's value for the same tree is not that number: compare trees on one file.
#
# The run is the PR's command — CUDA hidden the way tools/ref/dump.sh hides it (CUDA_VISIBLE_DEVICES=""),
# -ngl 0, -t 32, -c <ctx>, -b <batch>, --chunks N — plus --defer-experts, which skips the loader's
# MAP_POPULATE of a file set larger than the page cache and changes nothing the graph computes
# (router-trace.sh runs the same way). -ub is passed only when --ubatch is given. ik splits every
# batch into ubatches of n_ubatch, so --batch alone at or above n_ubatch runs the same graphs; --ubatch
# is the switch that changes them.
#
# The base file is ik's layout (examples/perplexity/perplexity.cpp; crates/gpu-gates/src/kld.rs reads it,
# `just gate-ds41-kld` checks it against its run): "_logits_", i32 n_ctx, n_vocab and n_chunk, the
# n_chunk*n_ctx ids as i32, then one record per scored position — ctx/2 .. ctx-2 of every chunk — of
# 2*((n_vocab+1)/2)+4 u16: f32 scale, f32 min_log_prob and a u16 level per vocabulary entry, about 1.06 GB
# at c2048 over 4 chunks [derived]. ik does not check its writes, so a base run writes <tag>.kld.part and
# renames it to <tag>.kld only when its length is the one its own header implies; before the lease it is
# refused (rc 2) when $BLOOMERY_DATA has less than twice the expected size free. A --kld run reads the
# base's header first — ik itself would warn and go on: a base of another ctx or chunk count than the
# run's, or the run's own tag, is refused (rc 64); a missing base, or one whose length disagrees with its
# header, rc 2.
#
# Refused before the lease, rc 3: a tree whose llama-perplexity, libllama.so or libggml.so is older than
# a file its working diff touches (`git diff --name-only HEAD`) — a stale build is a wrong number, not a
# missing one — and a binary that would load either library from outside the tree.
#
# The run takes the machine-wide CPU lease, prints a witness block before and after (the idiom of
# dump.sh and host-rate.sh), and is bounded by `timeout`: IK_PPL_BOUND seconds, default 1500. While it
# runs, the pid it runs under is in $BLOOMERY_DATA/ikppl/<tag>.pid (`timeout` passes a signal on);
# nothing here signals a pid found by a pattern. The last line is the result:
#
#   ppl tag=<tag> tree=<path> head=<rev> dirty_files=<n> model=<basename> ctx=<c> batch=<b> ubatch=<u> chunks=<N> ppl=<v> err=<±> per_chunk=[…] wall_s=<s>
#
# ubatch is the n_ubatch ik's log states. per_chunk lists what llama-perplexity prints after each chunk:
# the running estimate over chunks 1..i. wall_s is the tool's run, lease wait excluded. A base run adds
# kld=<path> kld_bytes=<n>
# kld_sha256=<first 12 hex> before wall_s. A --kld run prints no "Final estimate" line: its ppl and err
# are ik's "Mean PPL(Q)", per_chunk is the PPL column of its chunk table, and it adds base=<BASE> and
# ik's summary as printed — ppl_base, ln_ratio(_err) (the mean of ln PPL(Q)/PPL(base)), kld_mean(_err),
# kld_p99, kld_max, rms_dp(_err) and same_top(_err), the last two in percent.
set -euo pipefail
# MODEL= given to this command wins over the profile, which sets MODEL unconditionally.
CALLER_MODEL=${MODEL:-}
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
MODEL=${CALLER_MODEL:-$MODEL}

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" >&2; exit 64; }
[ $# -ge 2 ] || usage
TREE=$1
TAG=$2
shift 2
CHUNKS=4
CTX=2048
BATCH=
UBATCH=
KLD_BASE=0
KLD_OF=
while [ $# -gt 0 ]; do
  case $1 in
    --chunks) [ $# -ge 2 ] || usage; CHUNKS=$2; shift 2 ;;
    --ctx) [ $# -ge 2 ] || usage; CTX=$2; shift 2 ;;
    --batch) [ $# -ge 2 ] || usage; BATCH=$2; shift 2 ;;
    --ubatch) [ $# -ge 2 ] && [ -n "$2" ] || usage; UBATCH=$2; shift 2 ;;
    --kld-base) KLD_BASE=1; shift ;;
    --kld) [ $# -ge 2 ] && [ -n "$2" ] || usage; KLD_OF=$2; shift 2 ;;
    *) usage ;;
  esac
done
BATCH=${BATCH:-$CTX}
BOUND=${IK_PPL_BOUND:-1500}
# A leading zero is refused with the rest: bash arithmetic would read 0512 as octal.
for n in "$CHUNKS" "$CTX" "$BATCH" "$BOUND" ${UBATCH:+"$UBATCH"}; do
  case $n in ''|*[!0-9]*|0*) echo "ik-ppl.sh: --chunks, --ctx, --batch, --ubatch and IK_PPL_BOUND are positive integers, got '$n'" >&2; exit 64 ;; esac
done
if [ "$BATCH" -gt "$CTX" ]; then
  echo "ik-ppl.sh: --batch $BATCH is more than --ctx $CTX: ik would put several chunks in one batch, a mode this runner does not measure" >&2
  exit 64
fi
if [ -n "$UBATCH" ] && [ "$UBATCH" -gt "$BATCH" ]; then
  echo "ik-ppl.sh: --ubatch $UBATCH is more than --batch $BATCH: ik would clamp it to the batch without a word" >&2
  exit 64
fi
if [ "$KLD_BASE" = 1 ] && [ -n "$KLD_OF" ]; then
  echo "ik-ppl.sh: --kld-base writes a base and --kld reads one; a run does one of them" >&2
  exit 64
fi
for t in "$TAG" ${KLD_OF:+"$KLD_OF"}; do
  case $t in
    ''|*/*|.*|*[[:space:]]*) echo "ik-ppl.sh: '$t' cannot name a run" >&2; exit 64 ;;
  esac
done
if [ "$KLD_OF" = "$TAG" ]; then
  echo "ik-ppl.sh: --kld $KLD_OF is this run's own tag, and the run would overwrite the base's log" >&2
  exit 64
fi
# The default is the served V4.1 file; under another profile it would be another model, silently.
if [ -z "$CALLER_MODEL" ] && [ "$MODEL_NAME" != deepseek41 ]; then
  echo "ik-ppl.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  echo "  or name the file with MODEL=<gguf>" >&2
  exit 64
fi

TEXT=/home/user/eval/wikitext-2-raw/wiki.test.raw
[ -d "$TREE" ] || { echo "ik-ppl.sh: no tree at $TREE" >&2; exit 2; }
TREE=$(cd "$TREE" && pwd -P)
BIN=$TREE/build/bin/llama-perplexity
LIBLLAMA=$TREE/build/src/libllama.so
LIBGGML=$TREE/build/ggml/src/libggml.so
for f in "$BIN" "$LIBLLAMA" "$LIBGGML"; do
  [ -e "$f" ] || { echo "ik-ppl.sh: no $f — build the tree first" >&2; exit 2; }
done
[ -f "$MODEL" ] || { echo "ik-ppl.sh: no model at $MODEL" >&2; exit 2; }
[ -f "$TEXT" ] || { echo "ik-ppl.sh: no text at $TEXT" >&2; exit 2; }

# The ik trees belong to the serving user and this runs as root; safe.directory on the command line
# lets git read them without a config change.
g() { git -c safe.directory="$TREE" -C "$TREE" "$@"; }
HEAD_REV=$(g rev-parse --short HEAD)
DIRTY=$(g diff --name-only HEAD)
NDIRTY=$(printf '%s' "$DIRTY" | grep -c . || true)

oldest=$BIN
for f in "$LIBLLAMA" "$LIBGGML"; do
  if [ "$f" -ot "$oldest" ]; then oldest=$f; fi
done
newer=$(printf '%s\n' "$DIRTY" | while read -r f; do
          if [ -n "$f" ] && [ -e "$TREE/$f" ] && [ "$TREE/$f" -nt "$oldest" ]; then echo "$f"; fi
        done)
if [ -n "$newer" ]; then
  echo "[stale-build] $oldest is older than files the tree's diff touches:" >&2
  printf '%s\n' "$newer" | sed 's/^/    /' >&2
  echo "    rebuild the tree and rerun; measuring this build would be a wrong number, not a missing one." >&2
  exit 3
fi
for lib in libllama.so libggml.so; do
  got=$(ldd "$BIN" 2>/dev/null | awk -v l="$lib" '$1 == l { print $3 }' || true)
  case $(readlink -f "$got" 2>/dev/null) in
    "$TREE"/build/*) ;;
    *) echo "[foreign-lib] $BIN loads $lib from '${got:-nowhere}', not from $TREE/build" >&2; exit 3 ;;
  esac
done
sha() { sha256sum "$1" | cut -c1-12; }
BUILD_ID="llama-perplexity sha256=$(sha "$BIN") libllama.so sha256=$(sha "$LIBLLAMA") libggml.so sha256=$(sha "$LIBGGML")"

LOGDIR=$BLOOMERY_DATA/ikppl
mkdir -p "$LOGDIR"

# kld_header <file>: "<n_ctx> <n_vocab> <n_chunk>" from a base file's header; fails without ik's magic.
kld_header() {
  [ "$(head -c 8 -- "$1")" = _logits_ ] || return 1
  od -An -t d4 -j 8 -N 12 -- "$1" | awk '{ print $1, $2, $3 }'
}
# kld_bytes <n_ctx> <n_vocab> <n_chunk>: the length ik's layout gives those sizes.
kld_bytes() {
  echo $(( 20 + 4 * $3 * $1 + $3 * ($1 - 1 - $1 / 2) * 2 * (2 * (($2 + 1) / 2) + 4) ))
}
# The served V4.1 file's vocabulary (print_info: n_vocab), for the free-space estimate only: a finished
# base is checked against its own header.
KLD_VOCAB=129280
KLD=
BASEFILE=
mode=plain
if [ "$KLD_BASE" = 1 ]; then
  KLD=$LOGDIR/$TAG.kld
  mode="writes the base $KLD"
  want=$(kld_bytes "$CTX" "$KLD_VOCAB" "$CHUNKS")
  free=$(df -B1 --output=avail -- "$LOGDIR" | tail -n 1 | tr -d ' ')
  if [ "$free" -lt $((2 * want)) ]; then
    echo "ik-ppl.sh: $LOGDIR has $free bytes free, under twice the $want a base of ctx $CTX over $CHUNKS chunk(s) takes" >&2
    exit 2
  fi
elif [ -n "$KLD_OF" ]; then
  BASEFILE=$LOGDIR/$KLD_OF.kld
  mode="against the base $BASEFILE"
  [ -f "$BASEFILE" ] || { echo "ik-ppl.sh: no base file $BASEFILE — make it with --kld-base first" >&2; exit 2; }
  hdr=$(kld_header "$BASEFILE") || { echo "ik-ppl.sh: $BASEFILE does not open with ik's _logits_ magic" >&2; exit 2; }
  read -r b_ctx b_vocab b_chunks <<< "$hdr"
  if [ "$b_ctx" != "$CTX" ] || [ "$b_chunks" != "$CHUNKS" ]; then
    echo "ik-ppl.sh: $BASEFILE was made at ctx $b_ctx over $b_chunks chunk(s), this run is ctx $CTX over $CHUNKS: pass --ctx $b_ctx --chunks $b_chunks" >&2
    exit 64
  fi
  b_len=$(stat -c %s -- "$BASEFILE")
  b_want=$(kld_bytes "$b_ctx" "$b_vocab" "$b_chunks")
  if [ "$b_len" != "$b_want" ]; then
    echo "ik-ppl.sh: $BASEFILE is $b_len bytes; its header (ctx $b_ctx, n_vocab $b_vocab, $b_chunks chunk(s)) implies $b_want" >&2
    exit 2
  fi
fi

LOG=$LOGDIR/$TAG.log
OUT=$LOGDIR/$TAG.out
PIDFILE=$LOGDIR/$TAG.pid
: > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
dirty_list=$(printf '%s' "$DIRTY" | paste -sd' ' -)
say "ik-ppl.sh: tag $TAG, tree $TREE at $HEAD_REV, $NDIRTY changed file(s)${dirty_list:+: $dirty_list}, ctx $CTX, batch $BATCH, ubatch ${UBATCH:-ik default}, chunks $CHUNKS, $mode, bound ${BOUND}s"
say "  $BUILD_ID"
say "  model $MODEL"

# witness <tag>: the machine state the lease is supposed to guarantee. The device is the one the model
# file lives on; its sector count is machine-wide, so under the lease the difference between the two
# blocks is what this run paged in.
witness() {
  local dev sectors
  dev=$(df --output=source "$MODEL" 2>/dev/null | tail -n 1) || true
  sectors=$(awk '{print $3}' "/sys/class/block/${dev#/dev/}/stat" 2>/dev/null || echo '?')
  {
    echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch $(date +%s) ---"
    echo "loadavg: $(cat /proc/loadavg)"
    echo "pressure-cpu: $(grep '^some' /proc/pressure/cpu | head -n1)"
    echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
    echo "mem: $(grep -E '^(MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
    echo "pgmajfault: $(awk '$1 == "pgmajfault" {print $2}' /proc/vmstat)"
    echo "read-sectors: $sectors ($dev, 512 B each)"
    echo "cpu-mhz min/max: $(awk '$1 == "cpu" && $2 == "MHz" { if (lo == "" || $4 < lo) lo = $4; if ($4 > hi) hi = $4 } END { print lo, hi }' /proc/cpuinfo)"
    echo "gpu-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
    echo "lock-holder-pid: $$"
    echo "tree: $TREE head=$HEAD_REV dirty_files=$NDIRTY"
    echo "binary: $BUILD_ID"
    echo "model: $MODEL"
  } | tee -a "$LOG"
}

LOCK=/root/bloomery-cpu.lock
exec 9>"$LOCK"
say "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { say "[lease] timed out after 30 min"; exit 75; }
say "[lease] acquired $(date -u +%H:%M:%SZ)"
witness pre-ppl

MODE_ARGS=()
if [ -n "$UBATCH" ]; then
  MODE_ARGS=(-ub "$UBATCH")
fi
if [ "$KLD_BASE" = 1 ]; then
  MODE_ARGS+=(--kl-divergence-base "$KLD.part")
  # A run that fails, or a base whose length disagrees with its header, leaves no partial file behind.
  trap 'rm -f -- "$KLD.part"' EXIT
elif [ -n "$KLD_OF" ]; then
  MODE_ARGS+=(--kl-divergence --kl-divergence-base "$BASEFILE")
fi
t0=$(date +%s)
rc=0
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10 "$BOUND" \
  "$BIN" -m "$MODEL" -f "$TEXT" -c "$CTX" -b "$BATCH" --chunks "$CHUNKS" -ngl 0 -t 32 --defer-experts \
  ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
  > "$OUT" 2>> "$LOG" &
pid=$!
echo "$pid" > "$PIDFILE"
say "ik-ppl.sh: llama-perplexity under pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null || echo gone)), recorded in $PIDFILE"
wait "$pid" || rc=$?
t1=$(date +%s)
rm -f "$PIDFILE"
witness post-ppl
exec 9>&-
cat "$OUT" >> "$LOG"

if [ "$rc" != 0 ]; then
  say "ik-ppl.sh: llama-perplexity exited $rc$([ "$rc" = 124 ] && echo " — cut off at ${BOUND}s") after $((t1 - t0)) s; the log's tail:"
  tail -n 15 "$LOG" >&2
  exit "$rc"
fi
extra=
if [ -z "$KLD_OF" ]; then
  # "Final estimate: PPL over <n> chunks for n_ctx=<c> = <ppl> +/- <err>" — ik's wording of the last line,
  # with or without a base file.
  final=$(sed -n 's/^Final estimate: PPL over \([0-9]*\) chunks for n_ctx=\([0-9]*\) = \([0-9.]*\) +\/- \([0-9.]*\)$/\1 \2 \3 \4/p' "$OUT")
  read -r got_chunks got_ctx ppl err <<< "${final:-- - - -}"
  if [ "$got_chunks" != "$CHUNKS" ] || [ "$got_ctx" != "$CTX" ]; then
    say "ik-ppl.sh: no final estimate over $CHUNKS chunks at n_ctx=$CTX in $OUT (read: '${final:-nothing}')"
    exit 1
  fi
  per_chunk=$(grep -o '\[[0-9]*\][0-9.]*' "$OUT" | sed 's/^\[[0-9]*\]//' | paste -sd, - || true)
else
  # --kl-divergence prints a chunk table, rows "<i> <ppl> ± <u> <ln ratio> ± <u> <kld> ± <u> <rms Δp> ± <u> %
  # <same top> ± <u> %", then its summary: "Mean PPL(Q) : <v> ± <u>", "Mean    KLD: <v> ± <u>",
  # "Same top p: <v> ± <u> %" and so on (kl_divergence() in perplexity.cpp). ik exits 0 when it cannot read
  # the base, so a missing summary is the failure.
  per_chunk=$(awk 'NF == 18 && $1 ~ /^[0-9]+$/ && $3 == "±" { print $2 }' "$OUT" | paste -sd, - || true)
  summary=$(awk '
    $1 == "Mean" && $2 == "PPL(Q)"                 { ppl = $4; err = $6 }
    $1 == "Mean" && $2 == "PPL(base)"              { base = $4 }
    $1 == "Mean" && $2 == "ln(PPL(Q)/PPL(base))"   { lr = $4; lre = $6 }
    $1 == "Mean" && $2 == "KLD:"                   { k = $3; ke = $5 }
    $1 == "99.0%" && $2 == "KLD:"                  { k99 = $3 }
    $1 == "Maximum" && $2 == "KLD:"                { kmax = $3 }
    $1 == "RMS" && $3 == ":"                       { rms = $4; rmse = $6 }
    $1 == "Same" && $2 == "top" && $3 == "p:"      { top = $4; tope = $6 }
    END { if (ppl != "" && k != "" && tope != "") print ppl, err, base, lr, lre, k, ke, k99, kmax, rms, rmse, top, tope }
  ' "$OUT")
  read -r ppl err ppl_base lr lre km ke k99 kmax rms rmse top tope <<< "${summary:-}"
  n_rows=$(printf '%s' "$per_chunk" | tr ',' '\n' | grep -c . || true)
  if [ -z "${tope:-}" ] || [ "$n_rows" != "$CHUNKS" ]; then
    say "ik-ppl.sh: no KL-divergence summary over $CHUNKS chunks in $OUT (chunk rows $n_rows, summary '${summary:-nothing}')"
    exit 1
  fi
  extra=" base=$KLD_OF ppl_base=$ppl_base ln_ratio=$lr ln_ratio_err=$lre kld_mean=$km kld_err=$ke kld_p99=$k99 kld_max=$kmax rms_dp=$rms rms_dp_err=$rmse same_top=$top same_top_err=$tope"
fi
if [ "$KLD_BASE" = 1 ]; then
  hdr=$(kld_header "$KLD.part") || { say "ik-ppl.sh: $KLD.part does not open with ik's _logits_ magic"; exit 1; }
  read -r k_ctx k_vocab k_chunks <<< "$hdr"
  k_len=$(stat -c %s -- "$KLD.part")
  k_want=$(kld_bytes "$k_ctx" "$k_vocab" "$k_chunks")
  if [ "$k_ctx" != "$CTX" ] || [ "$k_chunks" != "$CHUNKS" ] || [ "$k_len" != "$k_want" ]; then
    say "ik-ppl.sh: $KLD.part is $k_len bytes with header ctx $k_ctx, n_vocab $k_vocab, $k_chunks chunk(s); that header implies $k_want, and this run is ctx $CTX over $CHUNKS"
    exit 1
  fi
  mv -f -- "$KLD.part" "$KLD"
  extra=" kld=$KLD kld_bytes=$k_len kld_sha256=$(sha "$KLD")"
fi
# "llama_init_from_model: n_ubatch      = 512": the physical batch ik ran, whether or not --ubatch set it.
ubatch=$(grep -a -m1 'n_ubatch' "$LOG" | awk '{ print $NF }' || true)
say "ppl tag=$TAG tree=$TREE head=$HEAD_REV dirty_files=$NDIRTY model=$(basename "$MODEL") ctx=$CTX batch=$BATCH ubatch=${ubatch:-?} chunks=$CHUNKS ppl=$ppl err=$err per_chunk=[$per_chunk]$extra wall_s=$((t1 - t0))"
