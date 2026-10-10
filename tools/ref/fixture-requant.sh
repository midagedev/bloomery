#!/usr/bin/env bash
# fixture-requant.sh FAMILY TAG — write the fixture variant TAG of FAMILY's fixture (`just fixture-requant FAMILY TAG`).
#
# A variant is the family's fixture requantized by ik's llama-quantize to the type map of TAG's row in tools/fixture-variants.tsv,
# every other tensor kept bit for bit: a quant type's train gate then runs on a small file instead of the family's download. It is
# written to fixture-variants/TAG/<fixture dir>/ beside the fixture root (tools/ref/ref-paths.sh owns the path), in the same shards as the fixture,
# named <fixture stem>-TAG-0000N-of-0000M.gguf, with the header key bloomery.fixture.variant = TAG and every other
# bloomery.fixture.* key of the fixture.
#
#   1. `fixture variant-rules` (crates/model/src/bin/fixture.rs) expands the row's map over the fixture's tensor names and prints
#      the quantizer's --custom-q argument, which names every tensor of the fixture at the type it takes (the map's at the new type,
#      each other at its own, which the quantizer copies) and the --override-kv that stamps the tag. A row that names no tensor, or
#      one already at the map's type, is refused there by name.
#   2. llama-quantize --allow-requantize --keep-split --leave-output-tensor writes into <dir>.tmp.<pid>.
#   3. `fixture variant-check` holds the written file to the fixture: every tensor's name, shape and shard equal, the type equal
#      but where the map names it (and then the map's), every other tensor's bytes equal, every bloomery.fixture.* key kept and
#      the tag added. Any other difference refuses by name and removes the temporary directory.
#   4. The directory is renamed into place; an existing <dir> is refused, as `fixture generate` refuses one, and nothing is
#      written over it.
#
# general.file_type and general.quantization_version are the quantizer's own words (its type argument and format version), not
# the file's content; the check does not hold them equal, and no reader of ours reads them.
#
# Needs BLOOMERY_TIER=fixture and no BLOOMERY_FIXTURE_VARIANT (the justfile recipe sets both); runs under the profile of FAMILY.
# REQUANT_THREADS (default 24) is the quantizer's thread count; REQUANT_BOUND (default 1500) its bound in seconds.
set -euo pipefail
HERE=${BASH_SOURCE[0]%/*}
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "fixture-requant.sh: $*" >&2; exit "${2:-64}"; }
[ $# = 2 ] || die "usage: fixture-requant.sh FAMILY TAG"
FAMILY=$1 TAG=$2
[ "${BLOOMERY_TIER:-real}" = fixture ] || die "BLOOMERY_TIER=fixture is required (the fixture is the source), got ${BLOOMERY_TIER:-real}"
[ -z "${BLOOMERY_FIXTURE_VARIANT:-}" ] || die "BLOOMERY_FIXTURE_VARIANT=$BLOOMERY_FIXTURE_VARIANT is set: the source is the family's own fixture, never a variant"
# tools/box.sh exports the profile it resolved as BLOOMERY_REF_MODEL_PROFILE (BLOOMERY_MODEL itself it does not export).
[ "${BLOOMERY_REF_MODEL_PROFILE:-$FAMILY}" = "$FAMILY" ] || die "the profile is '$BLOOMERY_REF_MODEL_PROFILE', not $FAMILY (BLOOMERY_MODEL=$FAMILY on the Mac side)"
BLOOMERY_MODEL=$FAMILY
case ${REQUANT_THREADS:-24} in '' | *[!0-9]* | 0*) die "REQUANT_THREADS is a positive integer" ;; esac
case ${REQUANT_BOUND:-1500} in '' | *[!0-9]* | 0*) die "REQUANT_BOUND is whole seconds >= 1" ;; esac

# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
[ -n "$FIXTURE_FILE" ] || die "the $FAMILY profile has no fixture file to requantize" 66
SRC=$FIXTURE_FILE
# The variant's directory, from the one owner of the path: the same file read for the tag, which also refuses a tag the table does
# not name or one of another family.
OUT=$(BLOOMERY_MODEL=$FAMILY FIXTURE_NEW=1 BLOOMERY_FIXTURE_VARIANT=$TAG bash -c '. "$1" && printf %s "$FIXTURE_DIR"' _ "$HERE/ref-paths.sh") ||
  die "ref-paths.sh refused the variant $TAG of $FAMILY" 64
[ -n "$OUT" ] || die "ref-paths.sh named no directory for the variant $TAG of $FAMILY" 64
[ ! -e "$OUT" ] || die "$OUT exists; a variant is never written over (remove it by hand to make it again)" 73
TMP=$OUT.tmp.$$
STEM=$(basename "$SRC" | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//')
[ "$STEM" != "$(basename "$SRC")" ] || die "$SRC is not a first shard named <stem>-0000N-of-0000M.gguf" 65
QUANT=$IK/build/bin/llama-quantize
[ -x "$QUANT" ] || die "no llama-quantize at $QUANT (the $FAMILY profile's ik tree)" 2

# Disk: the variant is smaller than its source; the source's bytes are the bound asked for.
NEED=$(find "$(dirname "$SRC")" -maxdepth 1 -name '*.gguf' -printf '%s\n' | awk '{ s += $1 } END { print s + 0 }')
mkdir -p "$(dirname "$OUT")"
FREE=$(df --output=avail -B1 "$(dirname "$OUT")" | tail -1 | tr -d ' ')
[ "$FREE" -gt "$NEED" ] || die "$FREE bytes free under $(dirname "$OUT"), the source is $NEED" 74

LOG=$REPO/target/fixture-requant-$TAG.log
mkdir -p "$REPO/target"
cleanup() {
  rc=$?
  if [ -d "$TMP" ]; then
    rm -rf "$TMP"
    echo "fixture-requant.sh: removed the incomplete $TMP (rc $rc)" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "fixture-requant.sh: $FAMILY $TAG: $SRC -> $OUT (source $NEED bytes, $FREE free)"
(cd "$REPO" && cargo build --release -p bloomery-model --bin fixture >&2)
FIXTURE_BIN=$REPO/target/release/fixture
TABLE=$REPO/tools/fixture-variants.tsv
RULES=$("$FIXTURE_BIN" variant-rules "$SRC" --tag "$TAG" --table "$TABLE") || die "fixture variant-rules refused $TAG" 65
CUSTOM_Q=$(awk -F'\t' '$1 == "custom-q" { print $2 }' <<< "$RULES")
OVERRIDE_KV=$(awk -F'\t' '$1 == "override-kv" { print $2 }' <<< "$RULES")
[ -n "$CUSTOM_Q" ] && [ -n "$OVERRIDE_KV" ] || die "fixture variant-rules printed no custom-q and override-kv lines" 65

mkdir "$TMP"
START=$(date +%s)
echo "fixture-requant.sh: quantizing (log $LOG, ${REQUANT_THREADS:-24} threads, bound ${REQUANT_BOUND:-1500} s)"
# The type argument is the quantizer's own and moves no tensor: every tensor has a rule. The output is the shards' prefix (--keep-split).
timeout --kill-after=10 "${REQUANT_BOUND:-1500}" "$QUANT" --allow-requantize --keep-split --leave-output-tensor \
  --custom-q "$CUSTOM_Q" --override-kv "$OVERRIDE_KV" "$SRC" "$TMP/$STEM-$TAG" Q4_K_M "${REQUANT_THREADS:-24}" > "$LOG" 2>&1 ||
  { tail -20 "$LOG" >&2; die "llama-quantize failed (log $LOG)" 70; }
echo "fixture-requant.sh: llama-quantize done in $(($(date +%s) - START)) s; log $LOG"
grep -c 'converting to' "$LOG" | xargs echo "fixture-requant.sh: tensors the quantizer converted:"

FIRSTS=("$TMP"/*-00001-of-*.gguf)
[ "${#FIRSTS[@]}" = 1 ] && [ -e "${FIRSTS[0]}" ] || die "$TMP holds no single first shard (${FIRSTS[*]})" 65
FIRST=${FIRSTS[0]}
"$FIXTURE_BIN" variant-check "$FIRST" --source "$SRC" --tag "$TAG" --table "$TABLE" > "$LOG.check" ||
  { tail -5 "$LOG.check" >&2; die "fixture variant-check refused the variant (log $LOG.check)" 65; }
tail -1 "$LOG.check"
sync "$TMP"/*.gguf
mv -T "$TMP" "$OUT"
echo "fixture-requant.sh: $OUT"
ls -l "$OUT"
du -sb "$OUT" | awk '{ print "fixture-requant.sh: variant bytes " $1 }'
