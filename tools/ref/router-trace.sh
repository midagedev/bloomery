#!/usr/bin/env bash
# router-trace.sh — trace ik_llama.cpp's MoE router over a token stream by prefill: every layer's
# top-k expert ids for every token, into $BLOOMERY_DATA/router/<name>/. tools/ref/router_trace.cpp
# says what it captures and what the set holds; tools/ref/router-coverage.py reads it.
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/router-trace.sh <corpus> [--name N] [args...]'
#   just trace-router <corpus> [--name N] [args...]            (deepseek41)
#   just trace-router-glm5next <corpus> [--name N] [args...]   (glm5next)
#   just trace-router-qwen4exp <corpus> [--name N] [args...]   (qwen4exp)
#
#   <corpus>   a name    the profile's corpus file: deepseek41's $BLOOMERY_DATA/engram/corpus-<corpus>.ids
#                        (engram-corpus.sh: code | prose | prose-all | ...), any other profile's
#                        $BLOOMERY_DATA/<profile>/corpus-<corpus>.ids, tokenized by its own vocabulary
#              oracle    the profile's REF_TOKENS at its REF_CTX, the oracle dump's input; the run
#                        ends with router-coverage.py comparing every layer's ids with the profile's
#                        oracle set, and exits with it
#              <path>    any ids file, one decimal token id per line
#   --name N   the set's directory under $BLOOMERY_DATA/router (default: the corpus name). Every
#              profile but deepseek41 names its sets `<profile>-<name>` (the default gets the prefix,
#              an explicit name must carry it): the directory is shared with V4.1's sets, whose
#              names came first. A set is only ever replaced by a trace of its own architecture.
#   --chunk C  tokens per independent context (default 2048; the oracle's REF_CTX for `oracle`). n_ctx
#              is C, except for `oracle`, which keeps the dump's REF_CTX.
#   args       passed to the harness after the defaults: --max-tokens N, --top-k-only (the fused
#              schedule, whose ids the oracle cannot check — router_trace.cpp), gpt_params such as
#              -ub (the fork dies at the end of a prefill at -ub 4096; stay at 2048 or below)
#
# The profile is picked on the Mac side: box.sh resolves the model under it, and a profile switch
# inside the box command is refused with exit 64 by ref-paths.sh.
#
# What the harness runs is the oracle dump's configuration — ik's CPU path with CUDA hidden, -ngl 0,
# -t 32 — plus --defer-experts. That flag skips the loader's MAP_POPULATE of the whole file set, about
# twice what this box's page cache holds, and changes nothing the graph computes; the prefill faults
# in what it touches. That is still most of the file set — most experts of every layer on every
# decode call — so the harness runs under the machine-wide CPU lease with witness blocks around it,
# the lease and witnesses of tools/ref/dump.sh.
#
# The harness is bounded by `timeout`: BLOOMERY_ROUTER_BOUND seconds, default 1800, so no trace runs
# past 30 minutes of box time unless the caller raised the bound on purpose. A run that is cut off
# installs nothing. The whole log goes to $BLOOMERY_DATA/router/<name>.log as well.
set -euo pipefail
# MODEL, IK, BLOOMERY_DATA, MODEL_NAME, REF_TOKENS, REF_CTX and REF_SET_CPU come from the profile.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
ROUTER=$BLOOMERY_DATA/router
BIN=$ROUTER/bin/router_trace
BOUND=${BLOOMERY_ROUTER_BOUND:-1800}

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" >&2; exit 64; }
[ $# -ge 1 ] || usage
CORPUS=$1
shift
NAME=
CHUNK=
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --name|--chunk)
      [ $# -ge 2 ] || usage
      if [ "$1" = --name ]; then NAME=$2; else CHUNK=$2; fi
      shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# V4.1's sets hold the directory's bare names; every other profile's carry its prefix.
if [ "$MODEL_NAME" = deepseek41 ]; then
  PREFIX= CORPORA=$BLOOMERY_DATA/engram
else
  PREFIX=$MODEL_NAME- CORPORA=$BLOOMERY_DATA/$MODEL_NAME
fi

mkdir -p "$ROUTER"
case $CORPUS in
  oracle)
    [ -n "${REF_TOKENS:-}" ] || { echo "router-trace.sh: the $MODEL_NAME profile sets no REF_TOKENS" >&2; exit 2; }
    IDS=$ROUTER/${PREFIX}oracle-tokens.ids
    tr ',' '\n' <<< "$REF_TOKENS" > "$IDS"
    CTX=$REF_CTX
    CHUNK=${CHUNK:-$REF_CTX} ;;
  */*)
    IDS=$CORPUS
    NAME=${NAME:-$PREFIX$(basename "$CORPUS" .ids)}
    CHUNK=${CHUNK:-2048}
    CTX=$CHUNK ;;
  *)
    IDS=$CORPORA/corpus-$CORPUS.ids
    CHUNK=${CHUNK:-2048}
    CTX=$CHUNK ;;
esac
NAME=${NAME:-$PREFIX$CORPUS}
[ -f "$IDS" ] || { echo "router-trace.sh: no ids file at $IDS (the profile's corpora: $CORPORA)" >&2; exit 2; }
case $NAME in
  bin|*/*|.*|*.staging|*.old|'') echo "router-trace.sh: '$NAME' cannot name a set" >&2; exit 64 ;;
esac
case $NAME in
  "$PREFIX"*) ;;
  *) echo "router-trace.sh: the $MODEL_NAME profile's sets are named ${PREFIX}<name>; '$NAME' is not" >&2; exit 64 ;;
esac
# A set of another architecture under this name is not this trace's to replace.
if [ -f "$ROUTER/$NAME/MANIFEST.tsv" ]; then
  HAS=$(awk -F'\t' '$1 == "# arch" { print $2; exit }' "$ROUTER/$NAME/MANIFEST.tsv")
  if [ "$HAS" != "$MODEL_NAME" ]; then
    echo "router-trace.sh: $ROUTER/$NAME holds a set of arch '${HAS:-none}', not $MODEL_NAME: pick another --name" >&2
    exit 64
  fi
fi

# Build first, outside the lease: the trace runs the source box.sh just synced, never an older binary.
bash "${BASH_SOURCE[0]%/*}/build-router-trace.sh"
[ -x "$BIN" ] || { echo "router-trace.sh: no harness at $BIN after the build" >&2; exit 2; }
# Which ik build the set is the output of, as dump.sh records it (the same git call): $IK is the tree
# the harness links. The tree belongs to the serving user, so root's git needs safe.directory.
ikgit() { git -c safe.directory='*' -C "$IK" "$@"; }
BUILD=$(ikgit rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
if [ "$BUILD" != unknown ] && ! ikgit diff --quiet HEAD 2>/dev/null; then BUILD="$BUILD-dirty"; fi
MD5=$(md5sum "$IDS" | cut -d' ' -f1)

LOG=$ROUTER/$NAME.log
exec > >(tee "$LOG") 2>&1
echo "router-trace.sh: corpus $CORPUS ($IDS, md5 $MD5) -> $ROUTER/$NAME, chunk $CHUNK, n_ctx $CTX, bound ${BOUND}s, ik $BUILD"

# dump.sh's witness fields: read-sectors is the model file's device, so the difference between the
# two blocks is what this trace paged in.
WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model)
lease_take
witness pre-trace

# The harness lives outside this tree's target/, where `just box-gc` looks, so the pid it runs
# under is written down at launch: `kill "$(cat $ROUTER/<name>.pid)"` stops a stuck trace (timeout
# passes the signal on to the harness). Nothing here signals a pid found by a pattern.
export CUDA_VISIBLE_DEVICES=""
PIDFILE=$ROUTER/$NAME.pid
rc=0
BLOOMERY_ROUTER_WRITE=1 BLOOMERY_REF_BUILD="$BUILD" BLOOMERY_ROUTER_IDS_MD5="$MD5" \
  timeout --kill-after=10 "$BOUND" \
  "$BIN" -m "$MODEL" --expect-arch "$MODEL_NAME" --ids "$IDS" --out "$ROUTER/$NAME" --chunk "$CHUNK" \
    -ngl 0 -c "$CTX" -t 32 --defer-experts "${ARGS[@]}" &
pid=$!
echo "$pid" > "$PIDFILE"
echo "router-trace.sh: harness under pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null || echo gone)), recorded in $PIDFILE"
wait "$pid" || rc=$?
rm -f "$PIDFILE"
witness post-trace
lease_release
if [ "$rc" != 0 ]; then
  echo "router-trace.sh: the harness exited $rc$([ "$rc" = 124 ] && echo " — cut off at ${BOUND}s") — nothing installed" >&2
  exit "$rc"
fi

SET=$ROUTER/$NAME
grep -q '^# complete' "$SET/MANIFEST.tsv" || { echo "router-trace.sh: $SET has no completion trailer" >&2; exit 1; }
grep -v -e '^layer' -e '^call' -e '^# layer' -e '^# call' "$SET/MANIFEST.tsv"
if [ "$CORPUS" = oracle ]; then
  python3 "${BASH_SOURCE[0]%/*}/router-coverage.py" oracle "$SET" "$BLOOMERY_DATA/$REF_SET_CPU"
fi
