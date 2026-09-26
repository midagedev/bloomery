#!/usr/bin/env bash
# Build the DSpark draft oracle on the box: an ik tree with the V4.1 draft rules applied, and
# dump_draft linked against it. Runs under tools/box.sh (as root); the ik trees belong to `user`,
# so every git, python and cmake step in them runs as that user.
#
# The tree is a git worktree of the V4.1 oracle tree (BASE, the sink-fixed db517b69) at that
# commit, detached, with the patch script applied and nothing else. The script exists only on the
# box (PATCH); it carries its own ROOT, which a copy with ROOT pointed at this tree replaces. The
# tree is checked, not trusted, on every run: its four patched files must be byte-identical to
# what the script makes of BASE_SHA's copies (rebuilt in a scratch directory from `git show`), and
# nothing else in the tree may differ from BASE_SHA. A tree that is neither pristine nor exactly
# patched is refused — a hand edit there would reach the oracle under this build's name.
#
# The cmake flags are the ones BASE's build was configured with (its CMakeCache's non-default
# entries). After the configure, every GGML_* and LLAMA_* cache entry is compared with BASE's
# cache, and a difference stops the build: the two trees must differ by the patch alone.
#
# The ik build uses every core for minutes, so it runs under the machine-wide CPU lease (a timed
# sitting would otherwise measure it). The oracle's `# build` line is DRAFT_BUILD below; the
# runner (dump-draft.sh) reads it from the file this script writes next to the binary.
#
# Output: $BLOOMERY_DATA/bin/dump_draft (DUMP_DRAFT_OUT moves it) and dump_draft.build beside it.
set -euo pipefail
BASE=/home/user/ik-idxkey
BASE_SHA=db517b69
PATCH=/home/user/ik-dsv41-draft.py
PATCH_SHA256=4eeb56aa1d75b470b556bab87ab90a467c8e4942b3269c9c15a3c78db0b1bc60
PATCHED=(src/llama-hparams.h src/llama-hparams.cpp src/llama-load-tensors.cpp src/graphs/build_deepseek4.cpp)
export IK=${DRAFT_IK:-/home/user/ik-dspark-draft}
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
# shellcheck source=tools/ref/lease.sh
source "$(dirname "${BASH_SOURCE[0]}")/lease.sh"
OUT=${DUMP_DRAFT_OUT:-$REF_BIN}
DRAFT_BUILD="$BASE_SHA+dsv41-draft ${PATCH_SHA256:0:12}"

as_user() { sudo -u user env HOME=/home/user PATH=/usr/local/cuda/bin:/usr/bin:/bin "$@"; }
ugit() { as_user git -C "$1" "${@:2}"; }

got=$(sha256sum "$PATCH" | cut -d' ' -f1)
[ "$got" = "$PATCH_SHA256" ] ||
  { echo "build-dump-draft: $PATCH has sha256 $got, this build names $PATCH_SHA256" >&2; exit 1; }
base_full=$(ugit "$BASE" rev-parse "$BASE_SHA^{commit}")

if [ ! -d "$IK" ]; then
  ugit "$BASE" worktree add --detach "$IK" "$BASE_SHA"
fi
head=$(ugit "$IK" rev-parse HEAD)
[ "$head" = "$base_full" ] || { echo "build-dump-draft: $IK is at $head, not $BASE_SHA ($base_full)" >&2; exit 1; }

# What the script makes of BASE_SHA's four files, in a scratch directory owned by user.
SCRATCH=$(as_user mktemp -d /tmp/dsv41-draft.XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT
for f in "${PATCHED[@]}"; do
  as_user mkdir -p "$SCRATCH/src/$(dirname "${f#src/}")"
  ugit "$IK" show "$BASE_SHA:$f" | as_user tee "$SCRATCH/$f" > /dev/null
done
as_user sed "s|^ROOT = .*|ROOT = \"$SCRATCH/src/\"|" "$PATCH" | as_user tee "$SCRATCH/patch.py" > /dev/null
grep -q "^ROOT = \"$SCRATCH/src/\"$" "$SCRATCH/patch.py" ||
  { echo "build-dump-draft: the ROOT line of $PATCH was not replaced" >&2; exit 1; }
as_user python3 "$SCRATCH/patch.py"

state=patched
for f in "${PATCHED[@]}"; do
  if cmp -s "$SCRATCH/$f" "$IK/$f"; then continue; fi
  if ugit "$IK" diff --quiet "$BASE_SHA" -- "$f"; then state=pristine; continue; fi
  echo "build-dump-draft: $IK/$f is neither $BASE_SHA's nor the patched copy — refusing" >&2
  exit 1
done
if [ "$state" = pristine ]; then
  for f in "${PATCHED[@]}"; do
    ugit "$IK" diff --quiet "$BASE_SHA" -- "$f" ||
      { echo "build-dump-draft: $IK is half patched ($f) — refusing" >&2; exit 1; }
    as_user cp "$SCRATCH/$f" "$IK/$f"
  done
  echo "build-dump-draft: applied the patch to $IK"
fi
# Nothing else may differ: the tracked diff is the four files, and no untracked file outside build/.
changed=$(ugit "$IK" diff --name-only "$BASE_SHA" | sort | tr '\n' ' ')
want=$(printf '%s\n' "${PATCHED[@]}" | sort | tr '\n' ' ')
[ "$changed" = "$want" ] || { echo "build-dump-draft: $IK differs from $BASE_SHA in [$changed], want [$want]" >&2; exit 1; }
untracked=$(ugit "$IK" status --porcelain --untracked-files=all | grep -v '^?? build/' | grep '^??' || true)
[ -z "$untracked" ] || { echo "build-dump-draft: untracked files in $IK:" >&2; echo "$untracked" >&2; exit 1; }
echo "build-dump-draft: $IK = $BASE_SHA + $PATCH (sha256 ${PATCH_SHA256:0:12})"
ugit "$IK" diff --stat "$BASE_SHA"

# BASE's configure, spelled out: the non-default entries of its CMakeCache.
as_user cmake -S "$IK" -B "$IK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /dev/null
cache_flags() { grep -E '^(GGML|LLAMA)_[A-Z0-9_]*:[A-Z]+=' "$1/build/CMakeCache.txt" | grep -v '_FOUND:' | sort; }
if ! diff <(cache_flags "$BASE") <(cache_flags "$IK"); then
  echo "build-dump-draft: $IK/build is configured differently from $BASE/build (diff above)" >&2
  exit 1
fi

lease_take
t0=$(date +%s)
# Bounded (BLOOMERY_BUILD_BOUND, default 1800 s, two gate bounds): the build holds every core under
# the lease, and a hung one must end rather than hold the machine. The bound runs as the tree's user,
# inside as_user, so it signals the build it started.
as_user timeout --kill-after=10 "${BLOOMERY_BUILD_BOUND:-1800}" cmake --build "$IK/build" -j "$(nproc)" --target llama common llama-spec-bench
echo "build-dump-draft: ik build in $(($(date +%s) - t0)) s"
mkdir -p "$OUT"
# A failed compile must not leave the previous binary for dump-draft.sh to run under this build's name.
rm -f "$OUT/dump_draft" "$OUT/dump_draft.build"
# -rdynamic exports dump_draft's ggml_backend_sched_graph_compute_async so libllama's calls bind to it
# (its header says why); -ldl for the dlsym(RTLD_NEXT) that forwards to libggml's.
ref_cxx -rdynamic -o "$OUT/dump_draft" "$HERE/tools/ref/dump_draft.cpp" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}" -ldl
lease_release
printf '%s\n' "$DRAFT_BUILD" > "$OUT/dump_draft.build"
echo "built $OUT/dump_draft against $IK ($DRAFT_BUILD)"
