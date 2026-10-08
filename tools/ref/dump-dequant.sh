#!/usr/bin/env bash
# dump-dequant.sh — the dequantization oracle's three sets, written once and read by `just gate-1-1`
# (and, for the synthetic blocks, the i-quant row gates). The writer behind `just dump-ref-dequant`;
# the readers open each set through its refset family (crates/refset/src/arch/dequant.rs), which
# refuse a set that another harness, another ggml library or another model file wrote.
#
# Runs under tools/box.sh (the toolchain env is sourced). It builds the harness (build-dequant.sh),
# then runs it three times:
#
#   $BLOOMERY_DATA/ref-dequant-v2lite         the default profile's file (V2-Lite), every type it holds
#   $BLOOMERY_DATA/ref-dequant-v41<suffix>    the V4.1 file the tree runs ($BLOOMERY_V41_MODEL), every
#                                             type it holds; the suffix is the profile's V41_SET_SUFFIX
#   $BLOOMERY_DATA/ref-synth                  `dequant_ref --synthetic`: the types no model file holds
#
# and writes DEQUANT.tsv into each, last: the harness executable and the libggml.so it loads (path,
# md5), the model file (or `(synthetic rows)`), one row per file dequant_ref wrote (bytes, md5) and
# the `# complete` trailer. The old DEQUANT.tsv goes first, so a set is unfinished while its files
# change.
#
# The two model sets have directories of their own: the `ref` and `ref-v41<suffix>` directories are
# where the gate before the identity file dumped them, and a checkout that still runs that gate
# rewrites them. `ref-synth` stays where it was, because the i-quant row gates read its `.blocks`;
# that gate rewrites the same bytes into it (the harness is deterministic), and its manifest.txt
# is written by rename.
#
# Not a measurement: no lease, no GPU, a few seconds.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/tools/ref/ref-paths.sh"
[ "$BLOOMERY_MODEL" = deepseek2 ] || {
  echo "dump-dequant: the V2-Lite set is the default profile's (deepseek2); BLOOMERY_MODEL=$BLOOMERY_MODEL would dump another file as it" >&2
  exit 64
}
V41=${BLOOMERY_V41_MODEL:?BLOOMERY_V41_MODEL unset — run through tools/box.sh, which exports it from the deepseek41 profile}
SUFFIX=$(. "$HERE/tools/ref/models/deepseek41.sh" && printf %s "$V41_SET_SUFFIX")

bash "$HERE/tools/ref/build-dequant.sh"
BIN=$BLOOMERY_DATA/bin/dequant_ref
LIBGGML=$(ldd "$BIN" | awk '$1 == "libggml.so" { print $3 }')
[ -f "$LIBGGML" ] || { echo "dump-dequant: $BIN loads no libggml.so (ldd names '$LIBGGML')" >&2; exit 66; }

md5() { md5sum < "$1" | cut -d' ' -f1; }

# dump <dir> <model> <dequant_ref argument>… — the harness into <dir>, then <dir>/DEQUANT.tsv.
dump() {
  local dir=$1 model=$2 types f n=0
  shift 2
  rm -f "$dir/DEQUANT.tsv"
  "$BIN" "$@"
  types=$(awk '{ print $2 }' "$dir/manifest.txt")
  {
    printf '# dequant_ref\t%s\t%s\n' "$BIN" "$(md5 "$BIN")"
    printf '# libggml\t%s\t%s\n' "$LIBGGML" "$(md5 "$LIBGGML")"
    printf '# model\t%s\n' "$model"
    printf 'file\tbytes\tmd5\n'
    for f in manifest.txt $(for t in $types; do for x in raw meta blocks; do [ -f "$dir/$t.$x" ] && echo "$t.$x"; done; done); do
      printf '%s\t%s\t%s\n' "$f" "$(wc -c < "$dir/$f")" "$(md5 "$dir/$f")"
      n=$((n + 1))
    done
    printf '# complete\t%s\n' "$n"
  } > "$dir/DEQUANT.tsv.tmp.$$"
  mv "$dir/DEQUANT.tsv.tmp.$$" "$dir/DEQUANT.tsv"
  head -4 "$dir/DEQUANT.tsv"
}

dump "$BLOOMERY_DATA/ref-dequant-v2lite" "$MODEL" "$MODEL" "$BLOOMERY_DATA/ref-dequant-v2lite"
dump "$BLOOMERY_DATA/ref-dequant-v41$SUFFIX" "$V41" "$V41" "$BLOOMERY_DATA/ref-dequant-v41$SUFFIX"
dump "$BLOOMERY_DATA/ref-synth" "(synthetic rows)" --synthetic "$BLOOMERY_DATA/ref-synth"
