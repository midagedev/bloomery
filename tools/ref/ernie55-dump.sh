#!/usr/bin/env bash
# Dump ERNIE 5.5 Flash Preview's oracle sets: the numpy reference (tools/ref/ernie55_ref.py: the checkpoint folder's forward in f32 on
# the CPU, one layer resident at a time, no llama.cpp, no transformers, no card) over the profile's folder, as dump_ref-format sets under
# $BLOOMERY_DATA:
#   ref_ernie55_short   34 ids     the opening clause of Alice in Wonderland: every layer's l_out, the routed layers' ids and weights,
#                                   result_norm, the logits before and after MultiMax, 8 greedy steps
#   ref_ernie55_long    2,328 ids  the chapter's first 23 paragraphs: the same, and each DSA layer's indexer_top_k. The one prompt past
#                                   the DSA indexer's threshold (its last position has 2,200 older-than-window positions for 2,048 kept)
# The rows and their shapes are the writer's (its docstring lists them). BLOOMERY_TIER=fixture runs the cut tools/fixture-cuts/ernie55.json
# makes (/models/fixtures/ernie55: the real file's layers 0, 1 and 7, the dense width 1,024 and the expert width 512) and installs the sets
# as fx_ref_ernie55_{short,long}.
#
# The ids are tools/ref/data/ernie55, made once by the writer's `tokenize` (the README there has the command): the dumper never tokenizes.
#
# Set names and identity are the family's (crates/refset/src/arch/ernie55): the sets' `# model` line is the folder's first shard; a cut's
# `# fixture` line is refset's (`refset-check --fixture-line` on that shard, built here the way tools/ref/ref-paths.sh's fixture_line builds
# it) and is passed to the writer, which refuses a cut without it and a line of another generation; their `# build` line names the writer by
# its md5, which the family pins.
#
# A set is installed only when both manifests carry their `# complete` trailer and the writer's `sanity` passes on the staged sets, after the
# run; the previous sets stay in place until then. A staging directory already there is refused by name, never overwritten, and one a dying
# run leaves is removed. The python is the system python3 with numpy from ERNIE55_PYLIB (default: the box's hfref venv, numpy 2.x OpenBLAS
# with `tokenizers`); the pool is ERNIE55_THREADS wide (default: the writer's).
#
# The run takes the machine lease (lease_take: BLOOMERY_LEASE_CARD names the prediction card), with CUDA hidden, under BLOOMERY_DUMP_BOUND
# (1800 s). DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk. REF_DUMP_LEASE=0 skips the lease for a
# reading that takes none.
set -euo pipefail
export BLOOMERY_MODEL=ernie5_5
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
# ref-paths.sh forms IKBIN from IK, and this profile has no ik tree (the reference is the numpy writer): the name is only so that
# line reads under set -u, as the Hy4 and MiniMax profiles' default IK is.
IK=${IK:-/no-ik-tree-for-ernie5_5}
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"

# The profile's model is the vendor's folder in the real tier and the cut in the fixture tier (ref-paths.sh refuses a tier that is
# neither, and a family with no fixture directory).
FOLDER=$MODEL
PYLIB=${ERNIE55_PYLIB:-/home/user/hfref-venv/lib/python3.12/site-packages}
REF=$HERE/tools/ref/ernie55_ref.py
IDS=$HERE/tools/ref/data/ernie55
SETS=(short long)
STEPS=8
STAGE=$BLOOMERY_DATA/ref-ernie55.staging
[ -f "$FOLDER/model.safetensors.index.json" ] || { echo "ernie55-dump.sh: no $FOLDER/model.safetensors.index.json (exit 66)" >&2; exit 66; }
FIRST=("$FOLDER"/*-00001-of-*.safetensors)
[ -f "${FIRST[0]}" ] || { echo "ernie55-dump.sh: no $FOLDER/*-00001-of-*.safetensors (exit 66)" >&2; exit 66; }
[ "${#FIRST[@]}" = 1 ] || { echo "ernie55-dump.sh: $FOLDER holds ${#FIRST[@]} first shards (${FIRST[*]}), not one (exit 66)" >&2; exit 66; }
# The witness's read-sectors field names the block device of MODEL: the folder's first shard.
# shellcheck disable=SC2034
MODEL=${FIRST[0]}
[ -d "$PYLIB/numpy" ] || { echo "ernie55-dump.sh: no numpy under $PYLIB (ERNIE55_PYLIB names a numpy 2.x OpenBLAS directory; exit 66)" >&2; exit 66; }
[ -f "$REF" ] || { echo "ernie55-dump.sh: no $REF (exit 66)" >&2; exit 66; }
for n in "${SETS[@]}"; do
  [ -f "$IDS/$n.ids" ] || { echo "ernie55-dump.sh: no $IDS/$n.ids (exit 66)" >&2; exit 66; }
done
THREADS=()
[ -z "${ERNIE55_THREADS:-}" ] || THREADS=(--threads "$ERNIE55_THREADS")

# A cut's `# fixture` line is refset's, taken from its owner: refset-check, built by ref-paths.sh's fixture_line (it prints the line, or
# ends 65 naming what it refused). The fixture tier's file is the cut (ref-paths.sh); the real tier has no line.
FIXTURE=()
if [ "${BLOOMERY_TIER:-real}" = fixture ]; then
  if [ "${DUMP_DRY:-0}" = 1 ]; then
    FIXTURE=(--fixture-line '<refset-check --fixture-line FIXTURE_FILE, taken when the run starts>')
  else
    # shellcheck disable=SC2311  # a failed build or refusal is the failure of this dump
    FIXTURE_LINE=$(BLOOMERY_TIER=fixture FIXTURE_FILE=${FIRST[0]} fixture_line ernie55-dump.sh)
    FIXTURE=(--fixture-line "$FIXTURE_LINE")
  fi
fi
PROMPTS=()
for n in "${SETS[@]}"; do PROMPTS+=(--prompt "$n=$IDS/$n.ids"); done

if [ "${DUMP_DRY:-0}" = 1 ]; then
  printf '[dry] tier\t%s\n[dry] folder\t%s\n[dry] stage\t%s\n[dry] pylib\t%s\n' "${BLOOMERY_TIER:-real}" "$FOLDER" "$STAGE" "$PYLIB"
  for n in "${SETS[@]}"; do printf '[dry] set\tref_ernie55_%s (installed under the fx_ prefix when the folder is a cut)\n' "$n"; done
  printf '[dry] argv\tpython3 -I %q run --model %q --out %q --pylib %q' "$REF" "$FOLDER" "$STAGE" "$PYLIB"
  printf ' %q' "${PROMPTS[@]}" --steps "$STEPS" ${THREADS[@]+"${THREADS[@]}"} ${FIXTURE[@]+"${FIXTURE[@]}"}
  printf '\n[dry] bound\t%s s\n[dry] lease\tnot taken (a dry run takes none)\n' "${BLOOMERY_DUMP_BOUND:-1800}"
  exit 0
fi
if [ -e "$STAGE" ]; then
  echo "ernie55-dump.sh: $STAGE exists: remove it by hand if it is yours to drop (exit 64)" >&2
  exit 64
fi

# The sets are installed by moving them out of the stage, so a stage still there at exit is a run that died.
drop_stage() {
  [ -d "$STAGE" ] || return 0
  rm -rf "$STAGE"
  echo "[ernie55] removed the incomplete $STAGE" >&2
}
trap drop_stage EXIT

dump_bound=${BLOOMERY_DUMP_BOUND:-1800}
# WITNESS is read by lease.sh's witness function.
# shellcheck disable=SC2034
WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder)
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then
  lease_take
  witness pre-dump
fi
export CUDA_VISIBLE_DEVICES=""
echo "[ernie55] folder $FOLDER stage $STAGE bound $dump_bound s"
lease_bounded "$dump_bound" python3 -I "$REF" run --model "$FOLDER" --out "$STAGE" --pylib "$PYLIB" \
  "${PROMPTS[@]}" --steps "$STEPS" ${THREADS[@]+"${THREADS[@]}"} ${FIXTURE[@]+"${FIXTURE[@]}"}

# Every set complete, or nothing is installed: a half of the family is never in place. The writer stages a set under its prompt's
# name; the installed name is the family's. Then the writer's own invariants of a written set.
SANITY=()
for n in "${SETS[@]}"; do
  grep -qs '^# complete' "$STAGE/$n/MANIFEST.tsv" ||
    { echo "ernie55-dump.sh: ref_ernie55_$n has no completion trailer: nothing installed (exit 1)" >&2; exit 1; }
  SANITY+=(--set "$STAGE/$n")
done
python3 -I "$REF" sanity --model "$FOLDER" --pylib "$PYLIB" "${SANITY[@]}"
prefix=
grep -qs '^# fixture' "$STAGE/${SETS[0]}/MANIFEST.tsv" && prefix=fx_
for n in "${SETS[@]}"; do
  dest=$BLOOMERY_DATA/${prefix}ref_ernie55_$n
  rm -rf "$dest.old"
  if [ -d "$dest" ]; then mv "$dest" "$dest.old"; fi
  mv "$STAGE/$n" "$dest"
  rm -rf "$dest.old"
  echo "installed $dest ($(grep -c '^tensor' "$dest/MANIFEST.tsv") tensors)"
done
rmdir "$STAGE"
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then witness post-dump; fi
echo "build: $(awk -F'\t' '$1 == "# build" { print $2; exit }' "$BLOOMERY_DATA/${prefix}ref_ernie55_${SETS[0]}/MANIFEST.tsv")  folder: $FOLDER"
