#!/usr/bin/env bash
# Produce the stage-1 oracle reference set: ik_llama.cpp's intermediate tensors for a
# fixed token sequence, as raw f32, into $BLOOMERY_DATA/ref/.
#
# The gate for rounds 1-2 through 1-5 reads what this writes. Run it once per model;
# re-run it whenever the ik build changes, because the reference is that build's output.
#
# The token sequence is the model profile's REF_TOKENS (tools/ref/models/<arch>.sh, picked by
# BLOOMERY_MODEL), not a prompt: the ids are a property of the model's tokenizer, read once
# with `$IK/build/bin/llama-tokenize -m $MODEL -p "The capital of France is" --ids
# --log-disable --no-parse-special` and pasted there (deepseek2's start with its BOS, 100000).
# BLOOMERY_REF_TOKENS overrides them. dump_ref does not tokenize on purpose — if it did, a
# tokenizer difference between it and bloomery would appear as a numeric difference in every
# downstream tensor and read as a kernel bug. Changing the ids invalidates the whole set.
#
# CUDA is switched off, not merely unused: with the CUDA backend registered, ik splits
# the graph (measured 2026-09-19: 351 splits with -ngl 0, 1 split with CUDA hidden) and
# the reference would then be a GPU reduction order that our CPU rounds cannot match.
#
# BLOOMERY_REF_BACKEND=cuda writes the GPU engine's oracle instead: the same dumper, the
# same tokens, every layer offloaded (-ngl 99) on the card the box env pins, into
# $BLOOMERY_DATA/ref_cuda/. The CPU set is never touched by that run. The GPU kernels use
# q8_1 activations like ik's CUDA path, so their band is against this set, not the CPU one.
#
# REF_DUMP_LEASE=1 runs the dump under the machine-wide CPU lease, the one the measure runners
# take, and brackets it with a witness block: wall time, bytes read from the model's block
# device, major faults, available memory and page cache before and after. The profile of a
# model whose dump pages in more than the machine can share sets it (deepseek41); for a
# profile that leaves it unset, the environment can. Do not wrap such a run in another flock
# on the same file: the second lock waits on the first for the full 30 minutes and ends rc 75.
set -euo pipefail
# MODEL, BLOOMERY_DATA and IK default in ref-paths.sh (BLOOMERY_REF_MODEL, BLOOMERY_DATA and IK
# override them).
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
BACKEND=${BLOOMERY_REF_BACKEND:-cpu}
case $BACKEND in
  cpu)  SET=$REF_SET_CPU;  NGL=0;  HIDE_CUDA=1 ;;
  cuda) SET=$REF_SET_CUDA; NGL=99; HIDE_CUDA=0 ;;
  *) echo "dump.sh: BLOOMERY_REF_BACKEND must be cpu or cuda, got '$BACKEND'" >&2; exit 2 ;;
esac
# Output-set override: the backend still picks the offload depth and CUDA visibility,
# BLOOMERY_REF_SET only renames the destination (e.g. ref_cuda_v2), so an instrumented
# dumper can produce a second set beside the one the gates read without touching it.
SET=${BLOOMERY_REF_SET:-$SET}
TOKENS=${BLOOMERY_REF_TOKENS:-${REF_TOKENS:-}}
[ -n "$TOKENS" ] || { echo "dump.sh: the $MODEL_NAME profile sets no REF_TOKENS" >&2; exit 2; }
LEASE=${REF_DUMP_LEASE:-0}
BIN="$BLOOMERY_DATA/bin/dump_ref"
[ -x "$BIN" ] || { echo "no dump_ref at $BIN — run: just build-ref-dump" >&2; exit 2; }

# witness <tag>: the machine state the lease is supposed to guarantee, as one block. The device
# is the one the model file lives on; its sector count is machine-wide, so under the lease the
# difference between the two blocks is what this dump paged in.
witness() {
  local dev sectors
  dev=$(df --output=source "$MODEL" 2>/dev/null | tail -n 1) || true
  sectors=$(awk '{print $3}' "/sys/class/block/${dev#/dev/}/stat" 2>/dev/null || echo '?')
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch $(date +%s) ---"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
  echo "mem: $(grep -E '^(MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  echo "pgmajfault: $(awk '$1 == "pgmajfault" {print $2}' /proc/vmstat)"
  echo "read-sectors: $sectors ($dev, 512 B each)"
  echo "lock-holder-pid: $$"
  echo "model: $MODEL_NAME"
}
if [ "$LEASE" = 1 ]; then
  LOCK=/root/bloomery-cpu.lock
  exec 9>"$LOCK"
  echo "[lease] waiting for $LOCK ..."
  flock -w 1800 9 || { echo "[lease] timed out after 30 min" >&2; exit 75; }
  echo "[lease] acquired $(date -u +%H:%M:%SZ)"
  witness pre-dump
fi

# Stage, then swap. The old set survives a failed run: `rm -f *.f32` up front used to mean
# that a dumper killed halfway left a half-set with nothing to compare it against, and on
# 2026-09-19 that is exactly what happened (a round ran the binary under gdb; every
# breakpoint killed it mid-write). A dump that does not finish must cost nothing.
REF="$BLOOMERY_DATA/$SET"
STAGE="$BLOOMERY_DATA/$SET.staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Which ik build this set is the output of — recorded in the manifest, because the
# reference is that build's answer and nothing else's.
# $IK comes from ref-paths.sh, the file build-dump.sh reads too — the tree this binary was
# linked against. Not $HOME/ik_llama.cpp: the dump runs as root and the tree is the serving user's.
BUILD=$(git -C "$IK" rev-parse --short HEAD 2>/dev/null || echo unknown)

if [ "$HIDE_CUDA" = 1 ]; then export CUDA_VISIBLE_DEVICES=""; fi
BLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR="$STAGE" BLOOMERY_REF_BUILD="$BUILD" \
  "$BIN" -m "$MODEL" --tokens "$TOKENS" -ngl "$NGL" -c "$REF_CTX" -t 32
if [ "$LEASE" = 1 ]; then witness post-dump; fi

# The trailer is the dumper's completion proof; without it the staged set is not installed.
grep -q '^# complete' "$STAGE/MANIFEST.tsv" || {
    echo "dump.sh: the staged set has no completion trailer — not installing it" >&2
    exit 1
}
# A set is one model's. The swap never installs a dump over a set another model file produced:
# a profile whose set name collided, or a stray REF_SET_CPU in the environment, would otherwise
# replace the set every gate of that other model reads. Compared by the basename of the
# `# model` line, which every manifest carries.
model_of() { awk -F'\t' '$1 == "# model" { n = split($2, p, "/"); print p[n]; exit }' "$1"; }
if [ -f "$REF/MANIFEST.tsv" ]; then
  was=$(model_of "$REF/MANIFEST.tsv")
  now=$(model_of "$STAGE/MANIFEST.tsv")
  if [ "$was" != "$now" ]; then
    echo "dump.sh: $REF holds a set of $was and this dump is of $now — not replacing it" >&2
    echo "  (the staged set stays in $STAGE; move the old set away or pick another BLOOMERY_REF_SET)" >&2
    exit 1
  fi
fi
rm -rf "$REF.old"
if [ -d "$REF" ]; then mv "$REF" "$REF.old"; fi
mv "$STAGE" "$REF"
rm -rf "$REF.old"
grep -c '^tensor' "$REF/MANIFEST.tsv" | xargs echo "reference tensors:"
echo "graph inputs: $(grep -c $'^input\t' "$REF/MANIFEST.tsv" || true)  integer twins: $(grep -c $'^int\t' "$REF/MANIFEST.tsv" || true)"
echo "build: $BUILD  backend: $BACKEND  set: $REF"
