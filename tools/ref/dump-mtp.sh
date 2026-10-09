#!/usr/bin/env bash
# Produce an MTP draft oracle set — GLM-5.3-Flash's or Qwen3.8-Flash-Next's, by the profile: every node ik computes on its MTP context while the
# target decodes a fixed prompt greedily with one MTP draft token a round, as raw f32, into
# $BLOOMERY_DATA/ref-mtp/<set>/ (dump_mtp.cpp's header describes the files and the manifest).
#
# ref-mtp is a root of its own: nothing here writes under the ref_* sets or ref-draft. Stage, then
# swap, as dump.sh does: the dump goes to <set>.staging, is installed only with its `# complete`
# trailer, and never replaces a set of another model file.
#
# The set: the first 64 ids of the glm5next profile's prose (GLM_PROSE, its sha256 checked), then 64
# positions decoded with ik's MTP stage at n_max = 1 (the draft width the MTP plan runs). Prose, not
# code: its accept rate (0.75 on this box, 09-15) leaves rejected rounds in the set, and code's (0.97)
# almost none. ik runs on the CPU with CUDA hidden, as the profile's node dumps do (dump.sh says why):
# the target, and the MTP context over the same model. No --defer-experts: the profile carries no
# REF_DUMP_ARGS, so the loader populates the split set. IK_PREGATE (an instrumentation read in the
# tree's target graph) is removed from the dumper's environment.
#
# The qwen4exp profile's set is the same loop over Qwen3.8: the first 64 ids of its prose (the d1k
# variant's ids file, models/qwen4exp.sh, sha256 checked), 64 positions at n_max = 1, the shared draft
# file beside the target (-md, QWEN38_MTP_DRAFT; ik's ctx_mtp runs build_qwen4exp's MTP graph over it and
# the target's token_embd and output), into ref-mtp/qwen4exp_prose64_n64_k1 (refset family mtp-qwen4exp).
# One dump_mtp serves both profiles: the tree the glm5next profile names (GLM_MTP_IK at GLM_MTP_SHA) is
# upstream's qwen4exp MTP graph with glm5next's merged on, and this script reads the two names from that
# profile, their one owner; build it with `just build-ref-dump-mtp`.
#
# The dump runs under the machine-wide CPU lease (32 threads, the whole split set paged in). Pick the
# profile on the Mac side: BLOOMERY_MODEL=glm5next|qwen4exp ./tools/box.sh 'bash tools/ref/dump-mtp.sh'.
#   MTP_SET_NAME     the set's name under ref-mtp (default prose64_n64_k1, qwen4exp_prose64_n64_k1)
#   REF_THREADS      ik's -t (default 32)
#   DUMP_MTP_OUT     the directory holding the dump_mtp to run (default $BLOOMERY_DATA/bin; build-dump-mtp.sh's own name)
#   DUMP_DRY         1: print what the dump would do (the set, `# model`, `# fixture`, `# draft_model`, `# build`, the dumper
#                    and its arguments) and stop with 0, after every refusal below (the stale-binary one included) and
#                    before the lease and anything on disk
#
# BLOOMERY_TIER=fixture dumps the same sets on the family's fixture file (`just dump-ref-mtp-fixture`): the model is its first
# shard (ref-paths.sh refuses any other), the set is its real twin's name behind `fx_` (SET_PREFIX, put on at one place below),
# and it carries the fixture file's `# fixture` line (refset-check --fixture-line, written verbatim by dump_mtp beside
# `# model`). The Qwen3.8 draft is then the file beside the fixture target under the real draft's name, which is where
# refset's fixture family looks for it; GLM's draft is the target itself, so its set has no `# draft_model` line. The real
# tier's dumps are as they were: no `# fixture` line, no prefix.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
case ${DUMP_DRY:-0} in
  0|1) ;;
  *) echo "dump-mtp.sh: DUMP_DRY is 1 (print the dump and stop) or 0 or unset, got '$DUMP_DRY'" >&2; exit 2 ;;
esac
fixture_dump_tier dump-mtp.sh
DRAFT_ARGS=()
case $MODEL_NAME in
  glm5next)
    MTP_IK=$GLM_MTP_IK MTP_SHA=$GLM_MTP_SHA
    TOKENS_FILE=$BLOOMERY_DATA/$GLM_PROSE TOKENS_SHA256=$GLM_PROSE_SHA256
    SET=${MTP_SET_NAME:-prose64_n64_k1} ;;
  qwen4exp)
    read -r MTP_IK MTP_SHA < <(bash -c 'source "$1" && printf "%s %s\n" "$GLM_MTP_IK" "$GLM_MTP_SHA"' _ \
      "${BASH_SOURCE[0]%/*}/models/glm5next.sh")
    [ -n "${MTP_IK:-}" ] && [ -n "${MTP_SHA:-}" ] ||
      { echo "dump-mtp.sh: models/glm5next.sh names no GLM_MTP_IK and GLM_MTP_SHA" >&2; exit 2; }
    ref_step_variant d1k || { echo "dump-mtp.sh: the qwen4exp profile has no d1k variant" >&2; exit 2; }
    TOKENS_FILE=$STEP_TOKENS_FILE TOKENS_SHA256=$STEP_TOKENS_SHA256
    # The refset family's DRAFT (crates/refset/src/arch/qwen4exp/mtp.rs): the set's `# draft_model` must name it.
    QWEN38_MTP_DRAFT=/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
    # The fixture's is beside the fixture target under that name (refset's fixture family: the target's directory, `{dir}/{name}`).
    [ -z "$SET_PREFIX" ] || QWEN38_MTP_DRAFT=${FIXTURE_FILE%/*}/${QWEN38_MTP_DRAFT##*/}
    [ -f "$QWEN38_MTP_DRAFT" ] || { echo "dump-mtp.sh: no draft file at $QWEN38_MTP_DRAFT" >&2; exit 2; }
    DRAFT_ARGS=(-md "$QWEN38_MTP_DRAFT")
    SET=${MTP_SET_NAME:-qwen4exp_prose64_n64_k1} ;;
  *) echo "dump-mtp.sh: the MTP oracle is GLM-5.3-Flash's or Qwen3.8-Flash-Next's; this command picked the" \
       "$MODEL_NAME profile" >&2; exit 2 ;;
esac
IK=$MTP_IK
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

N_PROMPT=64
N_PREDICT=64
SPEC=mtp:n_max=1
CTX=512
THREADS=${REF_THREADS:-32}
case $SET in
  */*|.*|*.staging|*.old|'') echo "dump-mtp.sh: '$SET' cannot name a set" >&2; exit 2 ;;
esac
SET=$SET_PREFIX$SET
case $THREADS in
  ''|*[!0-9]*|0*) echo "dump-mtp.sh: REF_THREADS must be a positive integer, got '$THREADS'" >&2; exit 2 ;;
esac

BIN=${DUMP_MTP_OUT:-$BLOOMERY_DATA/bin}/dump_mtp
[ -x "$BIN" ] || { echo "no dump_mtp at $BIN — run: just build-ref-dump-mtp" >&2; exit 2; }
# The binary must load ik from the MTP tree's build, be this tree's dump_mtp.cpp (by the sha256 its
# build record names; box.sh gives every file the box's "now", so mtimes cannot say), and be no older
# than the libraries it loads; the tree must still be GLM_MTP_SHA with nothing changed.
IK_REAL=$(readlink -f "$IK")
LIBS=()
for lib in libllama.so libggml.so; do
  got=$(ldd "$BIN" 2>/dev/null | awk -v l="$lib" '$1 == l { print $3 }' || true)
  case $(readlink -f "$got" 2>/dev/null) in
    "$IK_REAL"/build/*) LIBS+=("$(readlink -f "$got")") ;;
    *) echo "[foreign-lib] $BIN loads $lib from '${got:-nowhere}', not from $IK/build —" \
         "rebuild it: just build-ref-dump-mtp" >&2; exit 3 ;;
  esac
done
SRC="${BASH_SOURCE[0]%/*}/dump_mtp.cpp"
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
rec() { awk -v k="$1" '$1 == k { print $2 }' "$BIN.build" 2>/dev/null || true; }
stale=()
[ "$(rec source_sha256)" = "$SRC_SHA" ] || stale+=("$SRC (sha256 ${SRC_SHA:0:12}; the binary's record names '$(rec source_sha256 | cut -c1-12)')")
[ "$(rec build)" = "$MTP_SHA" ] || stale+=("the ik build (the record names '$(rec build)', the profile $MTP_SHA)")
for lib in "${LIBS[@]}"; do
  if [ "$lib" -nt "$BIN" ]; then stale+=("$lib (mtime $(date -u -r "$lib" +%Y-%m-%dT%H:%M:%SZ))"); fi
done
if [ ${#stale[@]} -gt 0 ]; then
  echo "[stale-binary] $BIN does not match:" >&2
  printf '    %s\n' "${stale[@]}" >&2
  echo "    rebuild it with just build-ref-dump-mtp" >&2
  exit 3
fi
# Read-only as root in the serving user's tree: no optional index refresh, so no root-owned .git file.
ikgit() { GIT_OPTIONAL_LOCKS=0 git -c safe.directory='*' -C "$IK" "$@"; }
head=$(ikgit rev-parse --short=8 HEAD)
if [ "$head" != "$MTP_SHA" ] || ! ikgit diff --quiet HEAD; then
  echo "dump-mtp.sh: $IK is at $head (or changed), not a clean $MTP_SHA — rebuild: just build-ref-dump-mtp" >&2
  exit 3
fi
BUILD=$MTP_SHA

[ -f "$TOKENS_FILE" ] || { echo "dump-mtp.sh: no ids file at $TOKENS_FILE" >&2; exit 2; }
got=$(sha256sum "$TOKENS_FILE" | cut -d' ' -f1)
[ "$got" = "$TOKENS_SHA256" ] ||
  { echo "dump-mtp.sh: $TOKENS_FILE has sha256 $got, the profile names $TOKENS_SHA256" >&2; exit 2; }

# What the dumper is handed, once: the fixture tier's `# fixture` line (empty in the real tier: no line, no argument; refset-check
# is built here, after the stale-binary refusal and before the lease) and the dumper's arguments. The dry run below prints these
# very values, so what it shows is what runs.
FIXTURE_LINE=$(fixture_line dump-mtp.sh)
FIXTURE_ARGS=()
[ -z "$FIXTURE_LINE" ] || FIXTURE_ARGS=(--fixture-line "$FIXTURE_LINE")
DUMP_ARGS=(-m "$MODEL" "${DRAFT_ARGS[@]}" --expect-arch "$MODEL_NAME" --tokens-file "$TOKENS_FILE"
  --tokens-count "$N_PROMPT" -n "$N_PREDICT" --spec-type "$SPEC" -ngl 0 -c "$CTX" -t "$THREADS" "${FIXTURE_ARGS[@]}")

# DUMP_DRY=1: print the dump and stop. Everything that refuses has run above; the lease, the staging directory and the dumper
# come below, and none of them may run in a dry run — this script takes the lease for every dump, so a dry exit that came after
# lease_take would hold the machine lease for a print.
if [ "${DUMP_DRY:-0}" = 1 ]; then
  printf '[dry] tier\t%s\tprofile %s\n' "${BLOOMERY_TIER:-real}" "$MODEL_NAME"
  printf '[dry] set\tref-mtp/%s\n' "$SET"
  printf '[dry] # model\t%s\n' "$MODEL"
  if [ -n "$FIXTURE_LINE" ]; then printf '[dry] %s\n' "$FIXTURE_LINE"; else echo "[dry] no # fixture line: the real tier writes none"; fi
  if [ ${#DRAFT_ARGS[@]} -gt 0 ]; then
    printf '[dry] # draft_model\t%s\n' "${DRAFT_ARGS[1]}"
  else
    echo "[dry] no # draft_model line: the target carries the NextN block"
  fi
  printf '[dry] # build\t%s\n' "$BUILD"
  printf '[dry] ik\t%s\n[dry] dumper\t%s\n' "$IK" "$BIN"
  shown=()
  for a in "${DUMP_ARGS[@]}"; do
    case $a in "# fixture"*) a='<the # fixture line>' ;; esac
    shown+=("$a")
  done
  printf '[dry] argv\t%q' "$BIN"
  printf ' %q' "${shown[@]}"
  printf '\n[dry] env\tBLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR=%s.staging BLOOMERY_REF_BUILD=%s BLOOMERY_REF_TOKENS_SHA256=%s\n' \
    "$BLOOMERY_DATA/ref-mtp/$SET" "$BUILD" "$TOKENS_SHA256"
  printf '[dry] lease\ttaken by every dump of this script (a dry run takes none)\n'
  exit 0
fi

WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model)
lease_take
witness pre-dump

ROOT=$BLOOMERY_DATA/ref-mtp
REF=$ROOT/$SET
STAGE=$ROOT/$SET.staging
mkdir -p "$ROOT"
rm -rf "$STAGE"
mkdir -p "$STAGE"
# A staged set without its trailer is removed on every exit, with a line naming it (dump.sh's rule).
drop_incomplete_stage() {
  [ -d "$STAGE" ] || return 0
  grep -qs '^# complete' "$STAGE/MANIFEST.tsv" && return 0
  rm -rf "$STAGE"
  echo "[staging] removed the incomplete $STAGE" >&2
}
trap drop_incomplete_stage EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
t0=$(date +%s)
# Bounded like dump.sh's dumper (BLOOMERY_DUMP_BOUND, default 1800 s): a hung dumper must end rather
# than hold the lease.
lease_bounded "${BLOOMERY_DUMP_BOUND:-1800}" env -u IK_PREGATE CUDA_VISIBLE_DEVICES= \
  BLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR="$STAGE" BLOOMERY_REF_BUILD="$BUILD" BLOOMERY_REF_TOKENS_SHA256="$TOKENS_SHA256" \
  "$BIN" "${DUMP_ARGS[@]}"
echo "dump-mtp.sh: dump in $(($(date +%s) - t0)) s"
witness post-dump

grep -q '^# complete' "$STAGE/MANIFEST.tsv" ||
  { echo "dump-mtp.sh: the staged set has no completion trailer — not installing it" >&2; exit 1; }
# A set names its model by the first shard's full path (`# model`) and, for Qwen3.8, its draft file
# (`# draft_model`); a set of other files is not replaced.
model_of() { awk -F'\t' '$1 == "# model" || $1 == "# draft_model" { print $2 }' "$1" | paste -sd' ' -; }
if [ -f "$REF/MANIFEST.tsv" ] && [ "$(model_of "$REF/MANIFEST.tsv")" != "$(model_of "$STAGE/MANIFEST.tsv")" ]; then
  echo "dump-mtp.sh: $REF holds a set of $(model_of "$REF/MANIFEST.tsv"), this dump is of" \
    "$(model_of "$STAGE/MANIFEST.tsv") — not replacing it (the staged set stays in $STAGE)" >&2
  exit 1
fi
rm -rf "$REF.old"
if [ -d "$REF" ]; then mv "$REF" "$REF.old"; fi
mv "$STAGE" "$REF"
rm -rf "$REF.old"
echo "mtp tensors: $(grep -c $'^tensor\t' "$REF/MANIFEST.tsv")  inputs: $(grep -c $'^input\t' "$REF/MANIFEST.tsv" || true)" \
  " blocks: $(grep -c $'^verify\t' "$REF/MANIFEST.tsv" || true)  files: $(find "$REF" -type f | wc -l)  bytes: $(du -sb "$REF" | cut -f1)"
grep $'^# blocks\t\|^# graphs\t' "$REF/MANIFEST.tsv"
echo "build: $BUILD  threads: $THREADS  set: $REF"
