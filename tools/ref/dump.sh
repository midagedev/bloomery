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
# The profile's name is the architecture the dumper expects (--expect-arch): it reads the file's
# header first and refuses another architecture before paging anything in, because the set name,
# the tokens and the lease below all come from the profile, not from the file.
#
# REF_DUMP_LEASE=1 runs the dump under the machine-wide CPU lease, the one the measure runners
# take, and brackets it with a witness block: wall time, bytes read from the model's block
# device, major faults, available memory and page cache before and after. The profile of a
# model whose dump pages in more than the machine can share sets it (deepseek41); for a
# profile that leaves it unset, the environment can. Do not wrap such a run in another flock
# on the same file: the second lock waits on the first for the full 30 minutes and ends rc 75.
# A profile's REF_DUMP_ARGS (an array) are added to every dump of that model.
#
#   dump.sh             the batch set: every token in one decode, into the profile's set
#   dump.sh <variant>   one decode step after a quiet prefill (dump_ref.cpp, --decode-step), into the
#                       set the profile's ref_step_variant names for it, with its context, ids and
#                       flags (models/deepseek41.sh lists them). CPU only; BLOOMERY_REF_SET still
#                       renames the destination, BLOOMERY_REF_TOKENS is refused.
#
# A decode-step set never takes the name of the profile's batch sets (REF_SET_CPU, REF_SET_CUDA),
# and the swap never replaces a set of the other kind, told apart by the `# prefill` header line
# only a decode step writes: a gate of the batch set would read a one-token graph, a gate of the
# step a batch one. A variant that reads its ids from a file names the file's sha256, and a file
# that no longer has it is refused before the lease.
set -euo pipefail
# MODEL, BLOOMERY_DATA and IK default in ref-paths.sh (BLOOMERY_REF_MODEL, BLOOMERY_DATA and IK
# override them).
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
BACKEND=${BLOOMERY_REF_BACKEND:-cpu}
case $BACKEND in
  cpu)  SET=$REF_SET_CPU;  NGL=0;  HIDE_CUDA=1 ;;
  cuda) SET=$REF_SET_CUDA; NGL=99; HIDE_CUDA=0 ;;
  *) echo "dump.sh: BLOOMERY_REF_BACKEND must be cpu or cuda, got '$BACKEND'" >&2; exit 2 ;;
esac
VARIANT=${1:-}
CTX=$REF_CTX
TOKENS_SHA256=
STEP_ARGS=()
if [ -z "$VARIANT" ]; then
  TOKENS=${BLOOMERY_REF_TOKENS:-${REF_TOKENS:-}}
  [ -n "$TOKENS" ] || { echo "dump.sh: the $MODEL_NAME profile sets no REF_TOKENS" >&2; exit 2; }
  TOKEN_ARGS=(--tokens "$TOKENS")
else
  declare -F ref_step_variant > /dev/null ||
    { echo "dump.sh: the $MODEL_NAME profile defines no decode-step variants" >&2; exit 2; }
  [ "$BACKEND" = cpu ] || { echo "dump.sh: decode-step variants are CPU sets, not $BACKEND" >&2; exit 2; }
  [ -z "${BLOOMERY_REF_TOKENS:-}" ] ||
    { echo "dump.sh: a variant carries its own ids; unset BLOOMERY_REF_TOKENS" >&2; exit 2; }
  ref_step_variant "$VARIANT" || { echo "dump.sh: the $MODEL_NAME profile has no variant '$VARIANT'" >&2; exit 2; }
  SET=$STEP_SET
  CTX=$STEP_CTX
  if [ -n "$STEP_TOKENS_FILE" ]; then
    [ -f "$STEP_TOKENS_FILE" ] || { echo "dump.sh: no ids file at $STEP_TOKENS_FILE" >&2; exit 2; }
    TOKENS_SHA256=$(sha256sum "$STEP_TOKENS_FILE" | cut -d' ' -f1)
    [ "$TOKENS_SHA256" = "$STEP_TOKENS_SHA256" ] || {
      echo "dump.sh: $STEP_TOKENS_FILE has sha256 $TOKENS_SHA256, the $VARIANT variant names $STEP_TOKENS_SHA256" >&2
      echo "  (the ids are what the set is of; update the profile only if the new file is meant)" >&2
      exit 2
    }
    TOKEN_ARGS=(--tokens-file "$STEP_TOKENS_FILE" --tokens-count $((STEP_PREFILL + 1)))
  else
    n=$(tr ',' '\n' <<< "$STEP_TOKENS" | grep -c . || true)
    [ "$n" = $((STEP_PREFILL + 1)) ] ||
      { echo "dump.sh: the $VARIANT variant lists $n ids for a prefill of $STEP_PREFILL" >&2; exit 2; }
    TOKEN_ARGS=(--tokens "$STEP_TOKENS")
  fi
  STEP_ARGS=(--decode-step "${STEP_ARGS[@]}")
fi
# Output-set override: the backend still picks the offload depth and CUDA visibility,
# BLOOMERY_REF_SET only renames the destination (e.g. ref_cuda_v2), so an instrumented
# dumper can produce a second set beside the one the gates read without touching it.
SET=${BLOOMERY_REF_SET:-$SET}
# REF_THREADS is ik's -t (default 32); any other count joins the set name as _t<N>, so no reader takes it for 32.
THREADS=${REF_THREADS:-32}
case $THREADS in
  ''|*[!0-9]*|0*) echo "dump.sh: REF_THREADS must be a positive integer, got '$THREADS'" >&2; exit 2 ;;
esac
if [ "$THREADS" != 32 ]; then SET=${SET}_t$THREADS; fi
case $SET in
  bin|*/*|.*|*.staging|*.old|'') echo "dump.sh: '$SET' cannot name a set" >&2; exit 2 ;;
esac
if [ -n "$VARIANT" ] && { [ "$SET" = "$REF_SET_CPU" ] || [ "$SET" = "$REF_SET_CUDA" ]; }; then
  echo "dump.sh: '$SET' is the $MODEL_NAME profile's batch set; a decode step goes into a set of its own" >&2
  exit 2
fi
if [ -n "$VARIANT" ]; then
  echo "dump.sh: $VARIANT — quiet prefill of $STEP_PREFILL, decode step at position $STEP_PREFILL, -c $CTX," \
    "into $SET; ${TOKEN_ARGS[*]}${TOKENS_SHA256:+ (sha256 $TOKENS_SHA256)} ${STEP_ARGS[*]}"
fi
LEASE=${REF_DUMP_LEASE:-0}
# The dump's bound, seconds: a hung dumper must end, under the lease or the gate lock alike. 1800 is
# five times the one V4.1 CPU dump on record, 356 s with the whole file set read cold (rig-log
# 2026-09-23); BLOOMERY_DUMP_BOUND overrides it for a longer set.
DUMP_BOUND=${BLOOMERY_DUMP_BOUND:-1800}
BIN="$BLOOMERY_DATA/bin/dump_ref"
[ -x "$BIN" ] || { echo "no dump_ref at $BIN — run: just build-ref-dump" >&2; exit 2; }
# One dump_ref serves every profile, and the manifest's `# build` names $IK: a binary linked against
# another ik tree would write that tree's answer under this tree's name. It must load both libraries
# from $IK's build.
IK_REAL=$(readlink -f "$IK")
LIBS=()
for lib in libllama.so libggml.so; do
  got=$(ldd "$BIN" 2>/dev/null | awk -v l="$lib" '$1 == l { print $3 }' || true)
  case $(readlink -f "$got" 2>/dev/null) in
    "$IK_REAL"/build/*) LIBS+=("$(readlink -f "$got")") ;;
    *) echo "[foreign-lib] $BIN loads $lib from '${got:-nowhere}', not from $IK/build —" \
         "rebuild it: IK=$IK bash tools/ref/build-dump.sh" >&2; exit 3 ;;
  esac
done
# The binary must be this tree's dump_ref.cpp, and no older than the ik libraries it loads. The
# source is compared by the sha256 build-dump.sh records beside the binary, not by mtime: tools/box.sh
# gives every file it transfers the box's own "now", so a fresh track tree is newer than any binary.
# timing-card.sh's assert_fresh_binary is not reused: it compares against crates/ sources, which
# dump_ref is not built from. Same wording and rc 3 as that refusal.
SRC="${BASH_SOURCE[0]%/*}/dump_ref.cpp"
BUILD_REC="$BIN.build"
BIN_SHA=$(sha256sum "$BIN" | cut -c1-12)
BIN_MTIME=$(date -u -r "$BIN" +%Y-%m-%dT%H:%M:%SZ)
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
REC_SHA=$(awk '$1 == "source_sha256" { print $2 }' "$BUILD_REC" 2>/dev/null || true)
stale=()
if [ -z "$REC_SHA" ]; then
  stale+=("$SRC: no build record at $BUILD_REC")
elif [ "$REC_SHA" != "$SRC_SHA" ]; then
  stale+=("$SRC (sha256 ${SRC_SHA:0:12}; the binary was built from ${REC_SHA:0:12})")
fi
for lib in "${LIBS[@]}"; do
  if [ "$lib" -nt "$BIN" ]; then stale+=("$lib (mtime $(date -u -r "$lib" +%Y-%m-%dT%H:%M:%SZ))"); fi
done
if [ ${#stale[@]} -gt 0 ]; then
  echo "[stale-binary] $BIN (sha256 $BIN_SHA, mtime $BIN_MTIME) is older than its sources:" >&2
  printf '    %s\n' "${stale[@]}" >&2
  echo "    rebuild it with IK=$IK bash tools/ref/build-dump.sh (just build-ref-dump) and rerun; a set" \
    "dumped by this one would be a wrong answer, not a missing one." >&2
  exit 3
fi
echo "[binary] $BIN sha256=$BIN_SHA mtime=$BIN_MTIME (dump_ref.cpp sha256 ${SRC_SHA:0:12})"

# The machine state the lease is supposed to guarantee: read-sectors is the model file's device, so
# the difference between the two blocks is what this dump paged in.
WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model)
if [ "$LEASE" = 1 ]; then
  lease_take
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
# A staged set without its completion trailer is removed on every exit (a failed dumper under
# `set -e`, a refusal below, a signal), with a line naming it: an incomplete set is never left
# behind for a reader to mistake for a set. A complete staged set that the swap refuses stays, as
# its refusal says. A SIGKILL skips the trap; the next dump of the set clears the directory above.
drop_incomplete_stage() {
  [ -d "$STAGE" ] || return 0
  grep -qs '^# complete' "$STAGE/MANIFEST.tsv" && return 0
  rm -rf "$STAGE"
  echo "[staging] removed the incomplete $STAGE" >&2
}
trap drop_incomplete_stage EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Which ik build this set is the output of — recorded in the manifest, because the
# reference is that build's answer and nothing else's.
# $IK comes from ref-paths.sh, the file build-dump.sh reads too — the tree this binary was
# linked against. Not $HOME/ik_llama.cpp: the dump runs as root and the tree is the serving user's.
# The tree belongs to the serving user and may be a worktree, so root's git needs safe.directory to
# read it; `-dirty` marks a build from uncommitted changes.
ikgit() { git -c safe.directory='*' -C "$IK" "$@"; }
BUILD=$(ikgit rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
if [ "$BUILD" != unknown ] && ! ikgit diff --quiet HEAD 2>/dev/null; then BUILD="$BUILD-dirty"; fi

if [ "$HIDE_CUDA" = 1 ]; then export CUDA_VISIBLE_DEVICES=""; fi
lease_bounded "$DUMP_BOUND" env BLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR="$STAGE" BLOOMERY_REF_BUILD="$BUILD" BLOOMERY_REF_TOKENS_SHA256="$TOKENS_SHA256" \
  "$BIN" -m "$MODEL" --expect-arch "$MODEL_NAME" "${TOKEN_ARGS[@]}" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
    "${REF_DUMP_ARGS[@]}" "${STEP_ARGS[@]}"
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
kind_of() { if grep -q $'^# prefill\t' "$1"; then echo decode-step; else echo batch; fi; }
if [ -f "$REF/MANIFEST.tsv" ]; then
  was=$(model_of "$REF/MANIFEST.tsv")
  now=$(model_of "$STAGE/MANIFEST.tsv")
  if [ "$was" != "$now" ]; then
    echo "dump.sh: $REF holds a set of $was and this dump is of $now — not replacing it" >&2
    echo "  (the staged set stays in $STAGE; move the old set away or pick another BLOOMERY_REF_SET)" >&2
    exit 1
  fi
  was=$(kind_of "$REF/MANIFEST.tsv")
  now=$(kind_of "$STAGE/MANIFEST.tsv")
  if [ "$was" != "$now" ]; then
    echo "dump.sh: $REF holds a $was set and this dump is a $now set — not replacing it" >&2
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
echo "build: $BUILD  backend: $BACKEND  threads: $THREADS  set: $REF"
if [ -n "$VARIANT" ]; then
  echo "decode step: variant $VARIANT, prefill $STEP_PREFILL, position $STEP_PREFILL, -c $CTX;" \
    "$(grep -c $'^skip-input\t.*\tgraph-scratch$' "$REF/MANIFEST.tsv" || true) graph-scratch leaves"
fi
