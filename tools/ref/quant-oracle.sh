#!/usr/bin/env bash
# shellcheck shell=bash
# The quant oracle on the box: tools/ref/quant_oracle.cpp linked against llama.cpp mainline's libggml-base and libggml-cpu
# (the Qwen tree, `just build-qvis-lcpp`: the commit the Qwen image-input sets name), and the set it writes.
#
#   bash tools/ref/quant-oracle.sh build   the harness into $BLOOMERY_DATA/bin/quant_oracle and its .build record
#   bash tools/ref/quant-oracle.sh dump    build, then the set $BLOOMERY_DATA/ref-quant, QUANT.tsv written last
#
# The set holds, per ggml weight type and row length, the quantized and random-code weight rows, their to_float, seeded activation
# rows with their vec_dot_type conversion, and every vec_dot (the harness's header lists the files). crates/refset/src/quant.rs
# reads it through the family of crates/refset/src/arch/quant.rs, which refuses by name a set whose harness (the md5 of the source
# here), mainline commit, model line or trailer is not its pin. QUANT.tsv states, around the harness's case and file rows:
#
#   # harness   the md5 of tools/ref/quant_oracle.cpp
#   # build     the first nine digits of the tree's commit (the commit's build: the md5 of one machine's library matches no other
#               build of it)
#   # cpu       the CPU features ggml's dot kernels were selected under (the sums depend on them)
#   # model     (synthetic rows): the rows are made by ggml's quantizer from a fixed seed, from no model file
#
# The tree is checked, not trusted: its HEAD is the pin, it has no tracked change, and the harness refuses a library that states
# another commit. Never rebuilt here. Not a measurement: no lease, no card (the CPU backend alone is linked).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/tools/ref/ref-paths.sh"

LCPP=${QUANT_LCPP:-/home/user/llama.cpp-36a73916}
PIN=36a73916ee0cb3b457f356066afabd47cce68884
OUT=${QUANT_ORACLE_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/quant_oracle.cpp
BIN=$OUT/quant_oracle
SET=$BLOOMERY_DATA/ref-quant
LIB=$LCPP/build/bin

die() {
  echo "quant-oracle: $1" >&2
  exit "${2:-2}"
}

md5() { md5sum < "$1" | cut -d' ' -f1; }

git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }

tree_check() {
  local have dirty l
  have=$(git_lcpp rev-parse HEAD) || die "$LCPP is not a git tree (just build-qvis-lcpp)"
  [ "$have" = "$PIN" ] || die "$LCPP is at $have, the oracle pins $PIN"
  dirty=$(git_lcpp status --porcelain --untracked-files=no) || die "cannot inspect tracked changes in $LCPP"
  [ -z "$dirty" ] || die "$LCPP has tracked changes; the oracle is the commit's build, not an edited tree"
  for l in libggml-base.so libggml-cpu.so; do
    [ -e "$LIB/$l" ] || die "no $LIB/$l (just build-qvis-lcpp)"
  done
  [ -f "$LCPP/build/CMakeCache.txt" ] || die "no $LCPP/build/CMakeCache.txt"
}

build() {
  local tmp source_md5 lcpp_path
  tree_check
  mkdir -p "$OUT"
  tmp=$OUT/quant_oracle.tmp.$$
  trap 'rm -f "$tmp" "$tmp.build"' EXIT
  g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -o "$tmp" "$SRC" -I"$LCPP/ggml/include" -I"$LCPP/ggml/src" \
    -L"$LIB" -lggml-cpu -lggml-base -Wl,-rpath,"$LIB"
  # Each fact is its own assignment: inside a printf's arguments a failed md5 or readlink would not end the script.
  source_md5=$(md5 "$SRC")
  lcpp_path=$(readlink -f "$LCPP")
  {
    printf 'source_md5 %s\nlcpp %s\nlcpp_commit %s\n' "$source_md5" "$lcpp_path" "$PIN"
    grep -E '^(GGML_NATIVE|GGML_CPU_ALL_VARIANTS|GGML_BACKEND_DL|CMAKE_BUILD_TYPE):' "$LCPP/build/CMakeCache.txt" | sed 's/^/cmake /'
  } > "$tmp.build"
  mv -f "$tmp" "$BIN"
  mv -f "$tmp.build" "$BIN.build"
  trap - EXIT
  echo "built $BIN from ${SRC##*/} md5 $source_md5 against $LCPP at ${PIN:0:9}"
}

# The harness's own files into QUANT.tsv, last: the old one goes first, so a set is unfinished while its files change. Each
# digest, size and the cpu line is read into a variable before it is printed, so a failed read ends the script, no trailer.
compose() {
  local f n=0 tmp=$SET/QUANT.tsv.tmp.$$ source_md5 cpu bytes digest
  trap 'rm -f "$tmp"' EXIT
  source_md5=$(md5 "$SRC")
  cpu=$(sed -n 's/^cpu\t//p' "$SET/info.tsv")
  {
    printf '# harness\t%s\n# build\t%s\n' "$source_md5" "${PIN:0:9}"
    printf '# cpu\t%s\n# model\t(synthetic rows)\n' "$cpu"
    cat "$SET/cases.tsv"
    printf '# file\tname\tbytes\tmd5\n'
    while read -r f; do
      bytes=$(wc -c < "$SET/$f" | tr -d ' ')
      digest=$(md5 "$SET/$f")
      printf 'file\t%s\t%s\t%s\n' "$f" "$bytes" "$digest"
      n=$((n + 1))
    done < "$SET/files.txt"
    printf '# complete\t%s\n' "$n"
  } > "$tmp"
  mv -f "$tmp" "$SET/QUANT.tsv"
  trap - EXIT
  rm -f "$SET/cases.tsv" "$SET/files.txt" "$SET/info.tsv"
  echo "wrote $SET/QUANT.tsv ($n files)"
}

dump() {
  build
  mkdir -p "$SET"
  rm -f "$SET/QUANT.tsv"
  find "$SET" -maxdepth 1 -type f \( -name '*.blocks' -o -name '*.act' -o -name '*.f32' -o -name '*.tsv' -o -name 'files.txt' \) -delete
  "$BIN" --pin "$PIN" --out "$SET"
  compose
}

case ${1:-} in
  build) build ;;
  dump) dump ;;
  *) die "usage: quant-oracle.sh build|dump" 64 ;;
esac
