#!/usr/bin/env bash
# Build the GLM-5.3-Flash MTP oracle on the box: dump_mtp linked against the ik tree that carries the
# glm5next MTP graph, the glm5next profile's GLM_MTP_IK at GLM_MTP_SHA (tools/ref/models/glm5next.sh).
# Runs under tools/box.sh (as root); the ik tree belongs to `user`, so every git and cmake step in it
# runs as that user.
#
# The tree is checked, not trusted: its HEAD must be GLM_MTP_SHA, with no tracked change and no
# untracked file outside what git ignores (its build/ and logs) — a hand edit there would reach the
# oracle under this build's name. The tree was built in place; `cmake --build` of the two libraries
# dump_mtp links then proves the build current against the sources (make finds nothing to do) or
# brings it up to date. It uses every core while it works, so it runs under the machine-wide CPU lease.
#
# Output: $BLOOMERY_DATA/bin/dump_mtp (DUMP_MTP_OUT moves it) and dump_mtp.build beside it, naming the
# sha256 of the dump_mtp.cpp it was built from, the ik tree and its build; dump-mtp.sh refuses a binary
# whose record does not match its own tree. The compile writes a temporary name and only a finished
# link replaces the installed binary.
set -euo pipefail
# The profile names the tree; IK must point at it before ref-build-common.sh derives the link flags.
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = glm5next ] ||
  { echo "build-dump-mtp: the MTP oracle is GLM-5.3-Flash's; this command picked the $MODEL_NAME profile" >&2; exit 2; }
export IK=$GLM_MTP_IK
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
# shellcheck source=tools/ref/lease.sh
source "$(dirname "${BASH_SOURCE[0]}")/lease.sh"
OUT=${DUMP_MTP_OUT:-$REF_BIN}
SRC=$HERE/tools/ref/dump_mtp.cpp

as_user() { sudo -u user env HOME=/home/user PATH=/usr/local/cuda/bin:/usr/bin:/bin "$@"; }
ugit() { as_user git -C "$IK" "$@"; }

head=$(ugit rev-parse --short=8 HEAD)
[ "$head" = "$GLM_MTP_SHA" ] ||
  { echo "build-dump-mtp: $IK is at $head, not $GLM_MTP_SHA" >&2; exit 1; }
ugit diff --quiet HEAD ||
  { echo "build-dump-mtp: $IK has tracked changes against $GLM_MTP_SHA:" >&2; ugit diff --stat HEAD >&2; exit 1; }
untracked=$(ugit status --porcelain --untracked-files=all | grep '^??' || true)
[ -z "$untracked" ] || { echo "build-dump-mtp: untracked files in $IK:" >&2; echo "$untracked" >&2; exit 1; }
[ -f "$IK/build/CMakeCache.txt" ] ||
  { echo "build-dump-mtp: $IK/build is not configured; this script builds in place, it does not configure" >&2; exit 1; }
echo "build-dump-mtp: $IK = $GLM_MTP_SHA, clean"

lease_take
t0=$(date +%s)
# Bounded (BLOOMERY_BUILD_BOUND, default 1800 s): a hung build must end rather than hold the lease. The
# bound runs as the tree's user, inside as_user, so it signals the build it started.
as_user timeout --kill-after=10 "${BLOOMERY_BUILD_BOUND:-1800}" cmake --build "$IK/build" -j "$(nproc)" --target llama common
echo "build-dump-mtp: ik libraries current in $(($(date +%s) - t0)) s"
mkdir -p "$OUT"
TMP=$OUT/dump_mtp.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
# -rdynamic exports dump_mtp's ggml_backend_sched_graph_compute_async and llama_set_mtp_op_type so
# libllama's calls bind to them (its header says why); -ldl for the dlsym(RTLD_NEXT) that forwards to
# the libraries' own.
ref_cxx -rdynamic -o "$TMP" "$SRC" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}" -ldl
lease_release
printf 'source_sha256 %s\nik %s\nbuild %s\n' "$SRC_SHA" "$(readlink -f "$IK")" "$GLM_MTP_SHA" > "$TMP.build"
mv -f "$TMP" "$OUT/dump_mtp"
mv -f "$TMP.build" "$OUT/dump_mtp.build"
echo "built $OUT/dump_mtp from dump_mtp.cpp sha256 ${SRC_SHA:0:12} against $IK ($GLM_MTP_SHA)"
