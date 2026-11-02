#!/usr/bin/env bash
# Mistral Large 4's DSpark oracle set, the one gate-gpu-mistrallarge4-dspark reads: upstream llama.cpp's own draft run
# on each line of tools/ref/ml4-dspark-prompts.txt (one prompt a line, `\n` for a newline inside it), from the tree
# `just build-ml4-dspark-lcpp` builds (llama.cpp at 785322d17 with tools/ref/ml4-dspark-dump.patch).
#
#   BLOOMERY_MODEL=mistral-large4 BLOOMERY_CARD=both tools/box.sh 'bash tools/ref/dump-ml4-dspark.sh [OUT]'
#
# Per prompt, llama-speculative-simple runs the profile's target (MODEL) with its draft (DSPARK) at --spec-type
# draft-dspark, --spec-draft-n-max N_MAX (the file's block, every row drafting), --spec-draft-p-min 0 (the file has no
# confidence head), greedy, 16 tokens, under LLAMA_DSPARK_DUMP=OUT/pNN: the patch writes the rows its draft is fed
# (taps.f32, taps.pos) and each call's ids (calls.tsv). Placement: the target's routed experts in host memory
# (-ot exps=CPU), the rest of it and the draft on the cards; the ids are greedy. The draft's KV cache stays f16 (the
# default): a quantized one is rotated before it is written, which the engine does not do.
# OUT defaults to $BLOOMERY_DATA/ref-draft/ml4_dspark (crates/refset/src/arch/mistrallarge4/dspark.rs SET); an existing
# OUT is refused (rc 65). MANIFEST.tsv, which the family's reader requires (crates/refset/src/lcppdspark.rs holds the
# format: the target, the draft, the tree's commit and the patch's md5, the architecture, N_MAX, one row a prompt, and
# the trailer), is written last. Not timed and no lease; each prompt's run is bounded at 900 s.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh"
[ "$MODEL_NAME" = mistral-large4 ] || { echo "dump-ml4-dspark: the profile is $MODEL_NAME, not mistral-large4" >&2; exit 64; }
TREE=/home/user/llama.cpp-ml4dspark
BIN=$TREE/build/bin/llama-speculative-simple
PATCH=$HERE/ml4-dspark-dump.patch
PROMPTS=$HERE/ml4-dspark-prompts.txt
OUT=${1:-$BLOOMERY_DATA/ref-draft/ml4_dspark}
N_MAX=3
[ -x "$BIN" ] || { echo "dump-ml4-dspark: no $BIN (just build-ml4-dspark-lcpp)" >&2; exit 2; }
[ -f "$DSPARK" ] || { echo "dump-ml4-dspark: no draft file $DSPARK" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "dump-ml4-dspark: $OUT exists; a set is written whole" >&2; exit 65; }
# The build the set names (crates/refset/src/arch/mistrallarge4/dspark.rs BUILD): the tree's commit, nine digits, and the
# patch's md5. Read before the first prompt, so a tree or a tool that cannot say ends the dump before it runs.
COMMIT=$(git -c safe.directory="$TREE" -C "$TREE" rev-parse HEAD | cut -c1-9) || { echo "dump-ml4-dspark: cannot read the commit of $TREE" >&2; exit 2; }
PATCH_MD5=$(md5sum < "$PATCH" | cut -d' ' -f1) || { echo "dump-ml4-dspark: cannot take the md5 of $PATCH" >&2; exit 2; }
[ "${#COMMIT}" = 9 ] && [ "${#PATCH_MD5}" = 32 ] || { echo "dump-ml4-dspark: commit [$COMMIT] and patch md5 [$PATCH_MD5] are not 9 and 32 digits" >&2; exit 2; }
mkdir -p "$OUT" || exit 2
n=0
rows=
while IFS= read -r line; do
  [ -n "$line" ] || continue
  prompt=$(printf '%b' "$line")
  d=$OUT/$(printf 'p%02d' "$n")
  mkdir -p "$d" || exit 2
  rc=0
  export LLAMA_DSPARK_DUMP=$d
  run_bounded timeout --kill-after=10 900 "$BIN" -m "$MODEL" -md "$DSPARK" --spec-type draft-dspark \
    --spec-draft-n-max "$N_MAX" --spec-draft-n-min 0 --spec-draft-p-min 0 -p "$prompt" -n 16 --temp 0 --seed 0 -c 4096 \
    -ngl 999 -ngld 999 -ot 'exps=CPU' < /dev/null > "$d/out.txt" 2> "$d/err.txt" || rc=$?
  [ "$rc" = 0 ] || { echo "dump-ml4-dspark: prompt $n ended rc $rc; its stderr is $d/err.txt" >&2; exit 1; }
  [ -s "$d/calls.tsv" ] && [ -s "$d/taps.pos" ] && [ -s "$d/taps.f32" ] \
    || { echo "dump-ml4-dspark: prompt $n wrote no dump in $d (is the tree patched?)" >&2; exit 1; }
  echo "dump-ml4-dspark: $d $(wc -l < "$d/calls.tsv") calls, first $(head -n 1 "$d/calls.tsv" | tr '\t' ' ')"
  rows+=$(printf 'prompt\t%s\t%s\t%s' "$(basename "$d")" "$(($(wc -c < "$d/taps.pos") / 4))" "$(($(wc -l < "$d/calls.tsv")))")$'\n'
  n=$((n + 1))
done < "$PROMPTS"
{
  printf '# model\t%s\n' "$MODEL"
  printf '# draft_model\t%s\n' "$DSPARK"
  printf '# build\t%s+patch-%s\n' "$COMMIT" "$PATCH_MD5"
  printf '# arch\t%s\n' "$MODEL_NAME"
  printf '# n_max\t%s\n' "$N_MAX"
  printf '# prompt columns\tname\tn_taps\tn_calls\n'
  printf '%s' "$rows"
  printf '# complete\t%s\t0\n' "$n"
} > "$OUT/MANIFEST.tsv"
echo "dump-ml4-dspark: $n prompts in $OUT"
