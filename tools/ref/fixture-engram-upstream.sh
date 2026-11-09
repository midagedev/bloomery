#!/usr/bin/env bash
# fixture-engram-upstream.sh — write the V4.1 fixture's upstream twin and its per-site counterpart
# (`just fixture-engram-upstream`).
#
# The twin is the deepseek41 fixture with its Engram in llama.cpp's layout (tools/ref/engram-relayout.py has the layout and
# the checks): one per_layer_token_embd table, the gains folded into engram_gate, engram_wkv as engram_kv, the hash under
# the ple.* keys. The table rows and the kv are written as Q6_K, transcoded exactly from the fixture's Q3_K (every value
# decodes to the same f32), so the twin has the public files' types and the fixture's values. The counterpart
# (`relayout --per-site`) keeps the fixture's per-site layout and carries the twin's numbers: the same Q6_K tables and wkv,
# engram_q the twin's gate, engram_k F32 ones. gate-gpu-ds41-engram-layout runs the two and holds them to one another.
#
#   1. engram-relayout.py twin names the directory from the fixture's (beside the variants, a name no variant tag takes);
#      an existing one is refused (73), as fixture-requant.sh refuses one. The counterpart lives in its per-site/.
#   2. engram-relayout.py relayout writes the twin into <dir>.tmp.<pid> and the counterpart into its per-site/, and holds
#      each to the fixture.
#   3. `fixture rebudget` records the card budget the family's planner gives each written file (their dense tensors are
#      not the fixture's bytes: the gains are F32, the kv Q6_K), as a variant's requant does.
#   4. The directory is renamed into place.
#
# Needs BLOOMERY_TIER=fixture and no BLOOMERY_FIXTURE_VARIANT (the justfile recipe sets both); runs under the deepseek41 profile.
set -euo pipefail
HERE=${BASH_SOURCE[0]%/*}
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "fixture-engram-upstream.sh: $*" >&2; exit "${2:-64}"; }
[ $# = 0 ] || die "usage: fixture-engram-upstream.sh"
[ "${BLOOMERY_TIER:-real}" = fixture ] || die "BLOOMERY_TIER=fixture is required (the fixture is the source), got ${BLOOMERY_TIER:-real}"
[ -z "${BLOOMERY_FIXTURE_VARIANT:-}" ] || die "BLOOMERY_FIXTURE_VARIANT=$BLOOMERY_FIXTURE_VARIANT is set: the source is the fixture itself"
[ "${BLOOMERY_REF_MODEL_PROFILE:-deepseek41}" = deepseek41 ] || die "the profile is '$BLOOMERY_REF_MODEL_PROFILE', not deepseek41"
BLOOMERY_MODEL=deepseek41
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
[ -n "$FIXTURE_FILE" ] || die "the deepseek41 profile has no fixture file" 66
OUT=$(python3 -I "$HERE/engram-relayout.py" twin "$FIXTURE_DIR") || die "engram-relayout.py named no twin directory for $FIXTURE_DIR"
[ ! -e "$OUT" ] || die "$OUT exists; a twin is never written over (remove it by hand to make it again)" 73
TMP=$OUT.tmp.$$
NEED=$(find "$FIXTURE_DIR" -maxdepth 1 -name '*.gguf' -printf '%s\n' | awk '{ s += $1 } END { print int(2 * s * 1.05) }')
mkdir -p "$(dirname "$OUT")"
FREE=$(df --output=avail -B1 "$(dirname "$OUT")" | tail -1 | tr -d ' ')
[ "$FREE" -gt "$NEED" ] || die "$FREE bytes free under $(dirname "$OUT"), the twin and its counterpart need about $NEED" 74
cleanup() {
  rc=$?
  if [ -d "$TMP" ]; then
    rm -rf "$TMP"
    echo "fixture-engram-upstream.sh: removed the incomplete $TMP (rc $rc)" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "fixture-engram-upstream.sh: $FIXTURE_FILE -> $OUT ($FREE bytes free)"
(cd "$REPO" && cargo build --release -p bloomery-model-hf --bin fixture >&2)
START=$(date +%s)
python3 -I "$HERE/engram-relayout.py" relayout "$FIXTURE_FILE" "$TMP" --table-type q6_K --kv-type q6_K ||
  die "engram-relayout.py refused the fixture" 65
python3 -I "$HERE/engram-relayout.py" relayout "$FIXTURE_FILE" "$TMP/per-site" --table-type q6_K --kv-type q6_K --per-site ||
  die "engram-relayout.py refused the fixture's per-site counterpart" 65
echo "fixture-engram-upstream.sh: relaid in $(($(date +%s) - START)) s"
for d in "$TMP" "$TMP/per-site"; do
  FIRSTS=("$d"/*-00001-of-*.gguf)
  [ "${#FIRSTS[@]}" = 1 ] && [ -e "${FIRSTS[0]}" ] || die "$d holds no single first shard (${FIRSTS[*]})" 65
  "$REPO/target/release/fixture" rebudget "${FIRSTS[0]}" || die "fixture rebudget refused ${FIRSTS[0]}: the card budget does not plan on it" 65
done
sync "$TMP"/*.gguf "$TMP"/per-site/*.gguf
mv -T "$TMP" "$OUT"
echo "fixture-engram-upstream.sh: $OUT"
ls -l "$OUT" "$OUT/per-site"
