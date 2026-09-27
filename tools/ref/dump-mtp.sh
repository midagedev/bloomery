#!/usr/bin/env bash
# Produce the GLM-5.3-Flash MTP draft oracle set: every node ik computes on its MTP context while the
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
# The dump runs under the machine-wide CPU lease (32 threads, the whole split set paged in). Pick the
# profile on the Mac side: BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/dump-mtp.sh'.
#   MTP_SET_NAME     the set's name under ref-mtp (default prose64_n64_k1)
#   REF_THREADS      ik's -t (default 32)
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = glm5next ] ||
  { echo "dump-mtp.sh: the MTP oracle is GLM-5.3-Flash's; this command picked the $MODEL_NAME profile" >&2; exit 2; }
IK=$GLM_MTP_IK
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

TOKENS_FILE=$BLOOMERY_DATA/$GLM_PROSE
TOKENS_SHA256=$GLM_PROSE_SHA256
N_PROMPT=64
N_PREDICT=64
SPEC=mtp:n_max=1
CTX=512
SET=${MTP_SET_NAME:-prose64_n64_k1}
THREADS=${REF_THREADS:-32}
case $SET in
  */*|.*|*.staging|*.old|'') echo "dump-mtp.sh: '$SET' cannot name a set" >&2; exit 2 ;;
esac
case $THREADS in
  ''|*[!0-9]*|0*) echo "dump-mtp.sh: REF_THREADS must be a positive integer, got '$THREADS'" >&2; exit 2 ;;
esac

BIN=$BLOOMERY_DATA/bin/dump_mtp
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
[ "$(rec build)" = "$GLM_MTP_SHA" ] || stale+=("the ik build (the record names '$(rec build)', the profile $GLM_MTP_SHA)")
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
if [ "$head" != "$GLM_MTP_SHA" ] || ! ikgit diff --quiet HEAD; then
  echo "dump-mtp.sh: $IK is at $head (or changed), not a clean $GLM_MTP_SHA — rebuild: just build-ref-dump-mtp" >&2
  exit 3
fi
BUILD=$GLM_MTP_SHA

[ -f "$TOKENS_FILE" ] || { echo "dump-mtp.sh: no ids file at $TOKENS_FILE" >&2; exit 2; }
got=$(sha256sum "$TOKENS_FILE" | cut -d' ' -f1)
[ "$got" = "$TOKENS_SHA256" ] ||
  { echo "dump-mtp.sh: $TOKENS_FILE has sha256 $got, the profile names $TOKENS_SHA256" >&2; exit 2; }

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
  "$BIN" -m "$MODEL" --expect-arch glm5next --tokens-file "$TOKENS_FILE" --tokens-count "$N_PROMPT" \
    -n "$N_PREDICT" --spec-type "$SPEC" -ngl 0 -c "$CTX" -t "$THREADS"
echo "dump-mtp.sh: dump in $(($(date +%s) - t0)) s"
witness post-dump

grep -q '^# complete' "$STAGE/MANIFEST.tsv" ||
  { echo "dump-mtp.sh: the staged set has no completion trailer — not installing it" >&2; exit 1; }
# A set names its model by the first shard's full path (`# model`); a set of another file is not replaced.
model_of() { awk -F'\t' '$1 == "# model" { print $2 }' "$1"; }
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
