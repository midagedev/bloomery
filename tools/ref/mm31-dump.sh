#!/usr/bin/env bash
# Dump the MiniMax-M3.1-Flash oracle sets: transformers' own MiniMaxM3VL decoder layers in f32 on the CPU over the vendor's
# safetensors folder (tools/ref/mm31/dump_tf.py), as the dump_ref-format sets the MiniMax-M3 gates read, under $BLOOMERY_DATA:
#   ref_minimaxm31_a          41 ids     every per-layer tap, 64 greedy tokens
#   ref_minimaxm31_b          55 ids     (Korean) each layer output l_out and the router's expert ids, 64 greedy tokens
#   ref_minimaxm31_c          116 ids    (Rust) the same as b
#   ref_minimaxm31_d          1,536 ids  of wikitext-2: the last position's logits only
#   ref_minimaxm31_e          2,560 ids  the sparse attention's block scores and selection under the vendor's rule (one top-16
#                                        over the index heads' maximum)
#   ref_minimaxm31_e_perhead  e's ids under llama.cpp's rule (one top-16 per index head), the selection the card kernel computes
# The ids are tools/ref/data/minimaxm3 (the two models share one tokenizer file); the taps and layouts are the M3 sets' (the
# header of dump_tf.py lists them). Set names and identity are the family's (crates/refset/src/arch/minimaxm31, family
# tf-minimaxm31): the sets' `# model` line is the folder's first shard and their `# build` line names the writer by its md5.
#
#   bash tools/ref/mm31-dump.sh prefill    layer-major prefill of every prompt; the staged state is what `greedy` continues from
#   bash tools/ref/mm31-dump.sh greedy     the token-major continuation of a, b, c; then every set is installed
#   bash tools/ref/mm31-dump.sh all        both in one job (the fixture cut: a few minutes)
#   bash tools/ref/mm31-dump.sh selftest   only the dequantization check against compressed-tensors and Fp8Dequantize, bit for bit
#
# prefill and greedy are two jobs, each under its own lease and bound (BLOOMERY_DUMP_BOUND, 1800 s): the staged directory
# $BLOOMERY_DATA/ref-minimaxm31.staging keeps the state between them. A job that ends at its bound or dies is continued by the
# same command with MM31_RESUME=1 (prefill from its last finished layer, greedy from its last finished step). A staged
# directory already there without MM31_RESUME=1 is refused by name, never overwritten.
#
# The folder is BLOOMERY_MM31_FOLDER, default /data/moved/models/MiniMax-M3.1-Flash: the dumper refuses a folder that is neither the
# revision it pins (by the index's sha256) nor a cut of that revision (the `bloomery.fixture.*` keys of the first shard,
# `dump_tf.py --kind` names which). A cut is the fixture tier: its `# fixture` line is refset's (`refset-check --fixture-line`
# on its first shard, built here the way tools/ref/ref-paths.sh's fixture_line builds it), given to the dumper, and its sets are
# installed under the `fx_` prefix, decided by the line the dumper wrote, never by this script's caller: a set of a cut cannot
# take a real name. BLOOMERY_TIER=fixture makes the default folder the cut tools/fixture-dirs.tsv names for minimax-m3 under
# BLOOMERY_FIXTURE_ROOT (default /models/fixtures), as ref-paths.sh's fixture tier resolves it (MODEL), and refuses the
# vendor's own folder.
#
# A set is installed only when every set's manifest carries its `# complete` trailer, after the last phase; the previous sets
# stay in place until then. The python is MM31_PY (default /home/user/hfref-venv/bin/python: torch, transformers,
# safetensors, numpy, compressed-tensors at the releases of tools/ref/mm31/requirements.txt). The CPU threads are MM31_THREADS
# (default: the dumper's, torch's).
#
# The run takes the machine lease (lease_take: BLOOMERY_LEASE_CARD names the prediction card, docs/cards/minimaxm31-dump.card),
# with CUDA hidden: nothing here uses a card. DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything
# on disk. REF_DUMP_LEASE=0 skips the lease for a dry reading on a Mac-side tree only.
set -euo pipefail
export BLOOMERY_MODEL=minimax-m3
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
case ${1:-} in
  prefill | greedy | all | selftest) PHASE=$1 ;;
  *) echo "mm31-dump.sh: prefill, greedy, all or selftest, got '${1:-}' (exit 64)" >&2; exit 64 ;;
esac
[ $# = 1 ] || { echo "mm31-dump.sh: one argument, got $# (exit 64)" >&2; exit 64; }
TIER=${BLOOMERY_TIER:-real}
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"

# The profile's model is the vendor's folder in the real tier and the cut in the fixture tier (ref-paths.sh refuses a tier that is
# neither, and a family with no fixture directory).
FOLDER=${BLOOMERY_MM31_FOLDER:-$MODEL}
PY=${MM31_PY:-/home/user/hfref-venv/bin/python}
DUMP=$HERE/tools/ref/mm31/dump_tf.py
SETS=(a b c d e e_perhead)
STAGE=$BLOOMERY_DATA/ref-minimaxm31.staging
[ -f "$FOLDER/model.safetensors.index.json" ] || { echo "mm31-dump.sh: no $FOLDER/model.safetensors.index.json (exit 66)" >&2; exit 66; }
FIRST=("$FOLDER"/*-00001-of-*.safetensors)
[ -f "${FIRST[0]}" ] || { echo "mm31-dump.sh: no $FOLDER/*-00001-of-*.safetensors (exit 66)" >&2; exit 66; }
[ "${#FIRST[@]}" = 1 ] || { echo "mm31-dump.sh: $FOLDER holds ${#FIRST[@]} first shards (${FIRST[*]}), not one (exit 66)" >&2; exit 66; }
# The witness's read-sectors field names the block device of MODEL: the folder's first shard, not the profile's GGUF.
# shellcheck disable=SC2034
MODEL=${FIRST[0]}
[ -x "$PY" ] || { echo "mm31-dump.sh: $PY is not executable (MM31_PY names a python with the requirements of tools/ref/mm31/requirements.txt; exit 66)" >&2; exit 66; }
[ -f "$DUMP" ] || { echo "mm31-dump.sh: no $DUMP (exit 66)" >&2; exit 66; }
THREADS=()
[ -z "${MM31_THREADS:-}" ] || THREADS=(--threads "$MM31_THREADS")
# The folder is the vendor's own or a cut of its revision, as the dumper reads the first shard; anything else it refuses by name.
KIND=$("$PY" -W ignore "$DUMP" --kind --folder "$FOLDER")
if [ "$TIER" = fixture ] && [ "$KIND" != cut ]; then
  echo "mm31-dump.sh: BLOOMERY_TIER=fixture runs on a cut, and $FOLDER is the vendor's own folder (exit 64)" >&2
  exit 64
fi

if [ "$PHASE" = selftest ]; then
  if [ "${DUMP_DRY:-0}" = 1 ]; then
    printf '[dry] argv\t%q' "$PY"
    printf ' %q' "$DUMP" --dequant-selftest --folder "$FOLDER" ${THREADS[@]+"${THREADS[@]}"}
    printf '\n'
    exit 0
  fi
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" -W ignore "$DUMP" --dequant-selftest --folder "$FOLDER" ${THREADS[@]+"${THREADS[@]}"}
fi

ARGS=(--folder "$FOLDER" --out "$STAGE" --phase "$PHASE" ${THREADS[@]+"${THREADS[@]}"})
RESUME=${MM31_RESUME:-0}
case $RESUME in
  0 | 1) ;;
  *) echo "mm31-dump.sh: MM31_RESUME is 0 or 1, got '$RESUME' (exit 64)" >&2; exit 64 ;;
esac
if [ "$RESUME" = 1 ]; then
  [ "$PHASE" != all ] || { echo "mm31-dump.sh: MM31_RESUME=1 continues prefill or greedy, not all (exit 64)" >&2; exit 64; }
  ARGS+=(--resume)
fi
# A cut's `# fixture` line is refset's, taken from its owner: refset-check, built by ref-paths.sh's fixture_line (it prints the
# line, or ends 65 naming what it refused). The vendor's own folder has none.
if [ "$KIND" = cut ]; then
  if [ "${DUMP_DRY:-0}" = 1 ]; then
    ARGS+=(--fixture-line '<refset-check --fixture-line FIRST_SHARD, taken when the run starts>')
  else
    FIXTURE_LINE=$(BLOOMERY_TIER=fixture FIXTURE_FILE=${FIRST[0]} fixture_line mm31-dump.sh)
    ARGS+=(--fixture-line "$FIXTURE_LINE")
  fi
fi
# What the staged directory must be for this phase: prefill and all start a new one (or continue one with MM31_RESUME=1); greedy
# continues the one prefill left, which the dumper itself refuses to find empty.
check_stage() {
  if [ "$PHASE" = greedy ]; then
    [ -d "$STAGE/_state" ] || { echo "mm31-dump.sh: greedy continues a prefill: no $STAGE/_state (run prefill first; exit 66)" >&2; exit 66; }
  elif [ "$RESUME" = 1 ]; then
    [ -d "$STAGE/_state" ] || { echo "mm31-dump.sh: MM31_RESUME=1 but no $STAGE/_state to continue (exit 66)" >&2; exit 66; }
  elif [ -e "$STAGE" ]; then
    echo "mm31-dump.sh: $STAGE exists: continue it with MM31_RESUME=1, or remove it by hand if it is yours to drop (exit 64)" >&2
    exit 64
  fi
}

if [ "${DUMP_DRY:-0}" = 1 ]; then
  printf '[dry] tier\t%s\n[dry] phase\t%s\tresume %s\n[dry] folder\t%s (%s)\n[dry] stage\t%s\n' "$TIER" "$PHASE" "$RESUME" "$FOLDER" "$KIND" "$STAGE"
  for n in "${SETS[@]}"; do printf '[dry] set\tref_minimaxm31_%s (installed under the fx_ prefix when the folder is a cut)\n' "$n"; done
  printf '[dry] argv\t%q' "$PY"
  printf ' %q' "$DUMP" "${ARGS[@]}"
  printf '\n[dry] bound\t%s s\n[dry] lease\tnot taken (a dry run takes none)\n' "${BLOOMERY_DUMP_BOUND:-1800}"
  exit 0
fi
check_stage

dump_bound=${BLOOMERY_DUMP_BOUND:-1800}
# WITNESS is read by lease.sh's witness function.
# shellcheck disable=SC2034
WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder)
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then
  lease_take
  witness "pre-$PHASE"
fi
export CUDA_VISIBLE_DEVICES=""
echo "[mm31] phase $PHASE folder $FOLDER stage $STAGE bound $dump_bound s"
lease_bounded "$dump_bound" "$PY" -W ignore "$DUMP" "${ARGS[@]}"
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then witness "post-$PHASE"; fi

if [ "$PHASE" = prefill ]; then
  for n in "${SETS[@]}"; do
    if grep -qs '^# complete' "$STAGE/ref_minimaxm31_$n/MANIFEST.tsv"; then echo "[mm31] ref_minimaxm31_$n complete"; else echo "[mm31] ref_minimaxm31_$n waits for greedy"; fi
  done
  echo "[mm31] prefill done: run 'greedy' next; nothing is installed yet"
  exit 0
fi

# Every set complete, or nothing is installed: a half of the family is never in place.
for n in "${SETS[@]}"; do
  grep -qs '^# complete' "$STAGE/ref_minimaxm31_$n/MANIFEST.tsv" ||
    { echo "mm31-dump.sh: ref_minimaxm31_$n has no completion trailer: nothing installed (exit 1)" >&2; exit 1; }
done
prefix=
grep -qs '^# fixture' "$STAGE/ref_minimaxm31_a/MANIFEST.tsv" && prefix=fx_
for n in "${SETS[@]}"; do
  dest=$BLOOMERY_DATA/${prefix}ref_minimaxm31_$n
  rm -rf "$dest.old"
  if [ -d "$dest" ]; then mv "$dest" "$dest.old"; fi
  mv "$STAGE/ref_minimaxm31_$n" "$dest"
  rm -rf "$dest.old"
  echo "installed $dest ($(grep -c '^tensor' "$dest/MANIFEST.tsv") tensors)"
done
rm -rf "$STAGE/_state"
rmdir "$STAGE"
echo "build: $(awk -F'\t' '$1 == "# build" { print $2; exit }' "$BLOOMERY_DATA/${prefix}ref_minimaxm31_a/MANIFEST.tsv")  folder: $FOLDER"
