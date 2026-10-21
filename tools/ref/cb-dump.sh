#!/usr/bin/env bash
# shellcheck shell=bash
# The build and the run of the mainline-API node dumper (tools/ref/cb_dump.cpp) against a family's llama.cpp tree: what the
# Solar-Open2 and MiniMax-M3 oracles share. Sourced by a family's <family>-dump.sh after ref-paths.sh and lease.sh; it only
# defines functions. The family sets, before calling cb_dump_build or cb_dump_run:
#   CB_FAMILY   the binary's and the sets' name: $BLOOMERY_DATA/bin/<family>_dump, the sets ref_<family>_<prompt>
#   CB_ENV      the prefix of its overrides: <ENV>_BIN (the binary), <ENV>_OUT (where the build puts it), <ENV>_THREADS
#   CB_SCRIPT   its script's file name, as the messages and fixture_dump_tier name it
#   CB_TREE     the llama.cpp tree the binary links (include/, ggml/include/, build/bin/libllama.so); not rebuilt here, since a
#               rebuild moves the libllama md5 the sets name
#   CB_PATCH    optional: the patch file a patched tree carries (its sha256 and CB_PATCH_TAG name it in the sets' `# build`);
#               unset for an unpatched tree, whose tracked files must then be clean and whose library md5 names the build
#   CB_ARCH     the general.architecture the file must state;  CB_TITLE  the manifest's first comment line
#   CB_IDS      the directory of the prompts' ids (tools/ref/data/<family>)
#   CB_PROMPTS  array of name:ids:nodes:greedy, nodes joined by `+` (the graph's cb() names)
#   CB_FLAGS    array of further dumper arguments (-b, -fa)
# Output of the build: $BLOOMERY_DATA/bin/<family>_dump and <family>_dump.build beside it, the record cb_dump_run compares: the
# source sha256 (cb_dump.cpp, ref_io.h, ref_tensor_io.h) and the tree's identity (cb_tree_record).
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)

# The tree's identity as `key value` lines: its commit, the patch's sha256 when it carries one, the md5 of the library the
# binary loads. A command that fails ends the function non-zero, so no record is written from an empty hash.
cb_tree_record() {
  local commit patch_sha='' lib_path lib_md5
  commit=$(git -c safe.directory='*' -C "$CB_TREE" rev-parse --short=9 HEAD) || return
  if [ -n "${CB_PATCH:-}" ]; then
    patch_sha=$(sha256sum "$CB_PATCH") || return
  fi
  lib_path=$(readlink -f "$CB_TREE/build/bin/libllama.so") || return
  lib_md5=$(md5sum < "$lib_path") || return
  printf 'tree_commit %s\n' "$commit"
  if [ -n "${CB_PATCH:-}" ]; then
    printf 'patch_sha256 %s\n' "${patch_sha%% *}"
  fi
  printf 'libllama_md5 %s\n' "${lib_md5%% *}"
}

# What the sets' `# build` line says: the commit, then the patch (`<commit>+<tag>-<12 hex of its sha256>`) or, for an unpatched
# tree, the library (`<commit>+libllama-<12 hex of its md5>`).
cb_tree_build() {
  local record commit sha
  record=$(cb_tree_record) || return
  commit=$(awk '$1 == "tree_commit" { print $2 }' <<< "$record")
  sha=$(awk '$1 == "patch_sha256" { print $2 }' <<< "$record")
  if [ -n "$sha" ]; then
    printf '%s+%s-%s\n' "$commit" "$CB_PATCH_TAG" "${sha:0:12}"
  else
    sha=$(awk '$1 == "libllama_md5" { print $2 }' <<< "$record")
    printf '%s+libllama-%s\n' "$commit" "${sha:0:12}"
  fi
}

# The sha256 of the dumper's source, the order the record keeps: the cpp, then the shared headers.
cb_source_sha() {
  cat "$HERE/tools/ref/cb_dump.cpp" "$HERE/tools/ref/ref_io.h" "$HERE/tools/ref/ref_tensor_io.h" | sha256sum | cut -d' ' -f1
}

cb_dump_build() {
  local out_var=${CB_ENV}_OUT out libdir src_sha tree dirty
  out=${!out_var:-$BLOOMERY_DATA/bin}
  libdir=$CB_TREE/build/bin
  [ -f "$libdir/libllama.so" ] || { echo "$CB_SCRIPT: no libllama.so in $libdir (the tree is built there)" >&2; exit 66; }
  if [ -n "${CB_PATCH:-}" ]; then
    [ -f "$CB_PATCH" ] || { echo "$CB_SCRIPT: no $CB_PATCH (the patch the tree carries)" >&2; exit 66; }
  else
    dirty=$(git -c safe.directory='*' -C "$CB_TREE" status --porcelain --untracked-files=no) ||
      { echo "$CB_SCRIPT: cannot inspect tracked changes in $CB_TREE" >&2; exit 2; }
    [ -z "$dirty" ] || { echo "$CB_SCRIPT: $CB_TREE has tracked changes; the oracle is the commit's build, not an edited tree" >&2; exit 2; }
  fi
  mkdir -p "$out"
  CB_TMP=$out/${CB_FAMILY}_dump.tmp.$$
  trap 'rm -f "$CB_TMP" "$CB_TMP.build"' EXIT
  src_sha=$(cb_source_sha) || { echo "$CB_SCRIPT: cannot hash the dumper's source" >&2; exit 2; }
  tree=$(cb_tree_record) || { echo "$CB_SCRIPT: cannot record $CB_TREE's identity (commit, patch, library)" >&2; exit 2; }
  g++ -std=c++17 -O2 -o "$CB_TMP" "$HERE/tools/ref/cb_dump.cpp" -I"$CB_TREE/include" -I"$CB_TREE/ggml/include" -I"$HERE/tools/ref" \
    -L"$libdir" -lllama -lggml -lggml-base -Wl,-rpath,"$libdir"
  printf 'source_sha256 %s\n%s\n' "$src_sha" "$tree" > "$CB_TMP.build"
  mv -f "$CB_TMP" "$out/${CB_FAMILY}_dump"
  mv -f "$CB_TMP.build" "$out/${CB_FAMILY}_dump.build"
  echo "built $out/${CB_FAMILY}_dump from cb_dump.cpp, ref_io.h and ref_tensor_io.h sha256 ${src_sha:0:12} against $CB_TREE at $(cb_tree_build)"
}

cb_dump_run() {
  local bin_var=${CB_ENV}_BIN thr_var=${CB_ENV}_THREADS BIN REC want tree build stage fixture_line_text dump_bound p name ids nodes greedy dest
  fixture_dump_tier "$CB_SCRIPT"
  BIN=${!bin_var:-$BLOOMERY_DATA/bin/${CB_FAMILY}_dump}
  REC=$BIN.build
  [ -x "$BIN" ] || { echo "$CB_SCRIPT: no $BIN — run: bash tools/ref/${CB_SCRIPT%.sh}-build.sh" >&2; exit 2; }
  want=$(cb_source_sha)
  [ "$(awk '$1 == "source_sha256" { print $2; exit }' "$REC")" = "$want" ] || {
    echo "[stale-binary] $BIN was linked from another cb_dump.cpp, ref_io.h or ref_tensor_io.h — run: bash tools/ref/${CB_SCRIPT%.sh}-build.sh" >&2
    exit 3
  }
  # The binary must still be the tree it was linked against: its commit, patch and library.
  tree=$(cb_tree_record) || { echo "$CB_SCRIPT: cannot record $CB_TREE's identity (commit, patch, library)" >&2; exit 2; }
  [ "$(awk '$1 != "source_sha256"' "$REC")" = "$tree" ] || {
    echo "[stale-binary] $BIN was linked against another build of $CB_TREE than it holds now ($(awk '$1 != "source_sha256" { printf "%s ", $2 }' "$REC")): rebuild it" >&2
    exit 3
  }
  build=$(cb_tree_build) || exit 2
  stage="$BLOOMERY_DATA/ref-$CB_FAMILY.staging"
  fixture_line_text=$(fixture_line "$CB_SCRIPT")
  # CB_ARCH and CB_TITLE are the family's, set before the call.
  # shellcheck disable=SC2153
  local args=(-m "$MODEL" --out "$stage" --build "$build" --arch "$CB_ARCH" --title "$CB_TITLE" -c 4096 -ub 512 -t "${!thr_var:-32}"
    ${CB_FLAGS[@]+"${CB_FLAGS[@]}"})
  [ -z "$fixture_line_text" ] || args+=(--fixture-line "$fixture_line_text")
  for p in "${CB_PROMPTS[@]}"; do
    IFS=: read -r name ids nodes greedy <<< "$p"
    args+=(--prompt "$name:$HERE/tools/ref/data/$CB_IDS/$ids.ids:$nodes:$greedy")
  done

  if [ "${DUMP_DRY:-0}" = 1 ]; then
    printf '[dry] tier\t%s\tprofile %s\n' "${BLOOMERY_TIER:-real}" "$MODEL_NAME"
    for p in "${CB_PROMPTS[@]}"; do printf '[dry] set\t%sref_%s_%s\n' "$SET_PREFIX" "$CB_FAMILY" "${p%%:*}"; done
    printf '[dry] # model\t%s\n[dry] # build\t%s\n[dry] dumper\t%s\n' "$MODEL" "$build" "$BIN"
    [ -z "$fixture_line_text" ] || printf '[dry] %s\n' "$fixture_line_text"
    printf '[dry] argv\t%q' "$BIN"
    printf ' %q' "${args[@]}"
    printf '\n[dry] lease\ttaken (a dry run takes none)\n'
    exit 0
  fi

  dump_bound=${BLOOMERY_DUMP_BOUND:-1800}
  # WITNESS is read by lease.sh's witness function.
  # shellcheck disable=SC2034
  WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model)
  if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then
    lease_take
    witness pre-dump
  fi
  STAGE=$stage
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  # The staged sets are removed on every exit unless each carries its completion trailer: a dump that does not finish costs nothing.
  # shellcheck disable=SC2329  # run by the EXIT trap below
  drop_incomplete_sets() {
    local d
    for d in "$STAGE"/*/; do
      [ -d "$d" ] || continue
      grep -qs '^# complete' "$d/MANIFEST.tsv" || { rm -rf "$d"; echo "[staging] removed the incomplete $d" >&2; }
    done
  }
  trap drop_incomplete_sets EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export CUDA_VISIBLE_DEVICES=""
  lease_bounded "$dump_bound" "$BIN" "${args[@]}"
  if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then witness post-dump; fi
  for p in "${CB_PROMPTS[@]}"; do
    name=${p%%:*}
    grep -q '^# complete' "$STAGE/$name/MANIFEST.tsv" || { echo "$CB_SCRIPT: $name has no completion trailer — not installing it" >&2; exit 1; }
  done
  for p in "${CB_PROMPTS[@]}"; do
    name=${p%%:*}
    dest="$BLOOMERY_DATA/${SET_PREFIX}ref_${CB_FAMILY}_$name"
    rm -rf "$dest.old"
    if [ -d "$dest" ]; then mv "$dest" "$dest.old"; fi
    mv "$STAGE/$name" "$dest"
    rm -rf "$dest.old"
    echo "installed $dest ($(grep -c '^tensor' "$dest/MANIFEST.tsv") tensors)"
  done
  rmdir "$STAGE"
  echo "build: $build  model: $MODEL"
}
