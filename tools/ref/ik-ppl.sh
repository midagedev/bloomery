#!/usr/bin/env bash
# ik-ppl.sh — wikitext-2 perplexity of one ik_llama.cpp tree on the V4.1 file, CPU only, under the
# machine-wide CPU lease. It is the measurement our ik port's PR quotes (c2048, 4 chunks, CPU only), so
# two trees run back to back through it are an A/B of the port.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-ppl.sh <tree> <tag> [--chunks N]'
#   just ik-ppl <tree> <tag> [--chunks N]
#
#   <tree>       an ik tree with build/bin/llama-perplexity built in it, e.g. /home/user/ik-idxkey
#   <tag>        names the run: $BLOOMERY_DATA/ikppl/<tag>.log (everything) and <tag>.out (the tool's
#                stdout), and the result line carries it
#   --chunks N   chunks to score (default 4, the PR's); 1 is a smoke run
#
# The model defaults to the profile's MODEL (tools/ref/models/deepseek41.sh), shard 1 of the served file
# set: the file the oracle and the engine read. A second 300+ GB file set would evict it from the page
# cache of this 256 GiB machine, and every V4.1 run after this one would page it back in. MODEL=<gguf> in
# the box command overrides it; from the Mac side BLOOMERY_REF_MODEL does (box.sh carries it over). The
# PR's own number was measured on the plain Q3_K_M file set, /models/DeepSeek-V4.1-Flash-Q3_K_M/, not on
# this graft, so this runner's value for the same tree is not that number: compare trees on one file.
#
# The run is the PR's command — CUDA hidden the way tools/ref/dump.sh hides it (CUDA_VISIBLE_DEVICES=""),
# -ngl 0, -t 32, -c 2048, -b 2048, --chunks N — plus --defer-experts, which skips the loader's
# MAP_POPULATE of a file set larger than the page cache and changes nothing the graph computes
# (router-trace.sh runs the same way).
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
#   ppl tag=<tag> tree=<path> head=<rev> dirty_files=<n> model=<basename> ctx=2048 chunks=<N> ppl=<v> err=<±> per_chunk=[…] wall_s=<s>
#
# per_chunk lists what llama-perplexity prints after each chunk: the running estimate over chunks 1..i.
# wall_s is the tool's run, lease wait excluded.
set -euo pipefail
# MODEL= given to this command wins over the profile, which sets MODEL unconditionally.
CALLER_MODEL=${MODEL:-}
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
MODEL=${CALLER_MODEL:-$MODEL}

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" >&2; exit 64; }
[ $# -ge 2 ] || usage
TREE=$1
TAG=$2
shift 2
CHUNKS=4
while [ $# -gt 0 ]; do
  case $1 in
    --chunks) [ $# -ge 2 ] || usage; CHUNKS=$2; shift 2 ;;
    *) usage ;;
  esac
done
BOUND=${IK_PPL_BOUND:-1500}
for n in "$CHUNKS" "$BOUND"; do
  case $n in ''|*[!0-9]*|0) echo "ik-ppl.sh: --chunks and IK_PPL_BOUND are positive integers, got '$n'" >&2; exit 64 ;; esac
done
case $TAG in
  ''|*/*|.*|*[[:space:]]*) echo "ik-ppl.sh: '$TAG' cannot name a run" >&2; exit 64 ;;
esac
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
LOG=$LOGDIR/$TAG.log
OUT=$LOGDIR/$TAG.out
PIDFILE=$LOGDIR/$TAG.pid
: > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
dirty_list=$(printf '%s' "$DIRTY" | paste -sd' ' -)
say "ik-ppl.sh: tag $TAG, tree $TREE at $HEAD_REV, $NDIRTY changed file(s)${dirty_list:+: $dirty_list}, chunks $CHUNKS, bound ${BOUND}s"
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

t0=$(date +%s)
rc=0
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10 "$BOUND" \
  "$BIN" -m "$MODEL" -f "$TEXT" -c 2048 -b 2048 --chunks "$CHUNKS" -ngl 0 -t 32 --defer-experts \
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
# "Final estimate: PPL over <n> chunks for n_ctx=<c> = <ppl> +/- <err>" — ik's wording of the last line.
final=$(sed -n 's/^Final estimate: PPL over \([0-9]*\) chunks for n_ctx=\([0-9]*\) = \([0-9.]*\) +\/- \([0-9.]*\)$/\1 \2 \3 \4/p' "$OUT")
read -r got_chunks got_ctx ppl err <<< "${final:-- - - -}"
if [ "$got_chunks" != "$CHUNKS" ] || [ "$got_ctx" != 2048 ]; then
  say "ik-ppl.sh: no final estimate over $CHUNKS chunks at n_ctx=2048 in $OUT (read: '${final:-nothing}')"
  exit 1
fi
per_chunk=$(grep -o '\[[0-9]*\][0-9.]*' "$OUT" | sed 's/^\[[0-9]*\]//' | paste -sd, - || true)
say "ppl tag=$TAG tree=$TREE head=$HEAD_REV dirty_files=$NDIRTY model=$(basename "$MODEL") ctx=2048 chunks=$CHUNKS ppl=$ppl err=$err per_chunk=[$per_chunk] wall_s=$((t1 - t0))"
