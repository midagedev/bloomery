#!/usr/bin/env bash
# `just metal-compare BASE`: the Metal backend's move proof (AGENTS.md's change-class table, the move row; `just ptx-compare`'s
# analogue on a Mac), in one command.
#   tools/metal-compare.sh BASE [--head REV | --here DIR] [--out DIR]
#
# What it proves. A change that should leave Qwen3.6's speed alone leaves the program alone, and the program of a Metal run is two
# things the MSL text does not show (every kernel is one translation unit, and the host's dispatch is outside the text):
#   1. the pipelines: `metal_scan` (crates/metal/src/bin/metal_scan.rs) builds every compute pipeline the engine builds, from both
#      GEMM programs, serializes each into a binary archive of its own and hashes it, so a kernel's hash is the device compiler's
#      output for that kernel on this Mac;
#   2. the host dispatch: `generate_metal --encode-trace` writes every dispatch (pipeline, grid, threads, every binding's buffer
#      by allocation order, offset and length, every `setBytes` value), barrier and commit/wait/event of a run, at depth ~0 (a
#      3-id prompt) and at depth 600 (a 600-id prompt, where the attention grid is the segment count of 600 keys). The generated ids
#      of both runs are compared as well.
# Both on the qwen35moe MLX split ($BLOOMERY_FIXTURE_ROOT/qwen35moe-mlxsplit). The scan and the trace need the `pipeline-scan` and
# `encode-trace` features of bloomery-metal; the shipped builds have neither, so their host code is the shipped code.
#
# The sides. BASE (any rev that names a commit) is read through `git archive` into $METAL_COMPARE_WORK/<tree>/<sha12> (default
# ~/.cache/bloomery/metal-compare; kept, so a second run builds incrementally; delete it to clear), never the working tree. The head
# side is this tree in place, uncommitted files included; `--head REV` archives a commit instead, `--here DIR` takes a directory.
# A side whose crates/metal has no instrument (a commit older than the one that added crates/metal/src/trace.rs) gets it laid over:
# the diff of that commit against its parent (this tree's uncommitted crates/metal against HEAD while the instrument is not
# committed) applied with `patch`; a side it does not apply to is refused by name (rc 2), and a `--here` directory is never patched.
# Each side builds in a target directory of its own under $METAL_COMPARE_TARGET (default target/metal-compare/<dir>-<md5 of its
# path>, ~200 MB, a cold build ~25 s): cargo gives the same workspace crate in two trees one metadata hash, so one shared directory
# lets a build script's output (the kernels' source) of one tree stand in for the other's. Their binaries are kept in OUT/<side>/.
#
# The scan's one instability. The device compiler's archive of a function carries a few small numbers that move with the compiler's
# session state: all 115 rows change at once and no archive changes size. Seen on one Mac: two states, one holding for tens of
# minutes, and one run in the other state seconds after a run in the first. The state holds inside one process (the scan's twin
# build agrees with it), so
# a scan is never compared with one from another sitting, and each side is scanned twice, in two processes next to each other; a
# side that does not reproduce its own scan, or a pair that differs in every hash and nothing else, is scanned again (three attempts).
#
# Output (OUT, default target/metal-compare-out/<base12>-<head label>): base/ and head/ each hold metal_scan, generate_metal,
# build.log, pipelines.tsv and pipelines-2.tsv, trace-d0.txt, trace-d600.txt, ids-*.txt and the runs' logs; compare.txt is the
# printed tables.
#
# Exit codes: 0 every pipeline row and both traces and both id lists identical; 3 an existing pipeline moved or a trace or an id
# list differs; 4 a pipeline was removed; 5 both; 6 nothing existing moved but the head builds pipelines the base does not (the add
# class: its rows are listed); 2 a side refused by name (no instrument and none applies, two scans of different environments, a
# file that is no scan or trace); 1 a build or a run failed (named, with its log); 64 usage; 78 not an Apple Silicon Mac.
# Runs no gate and takes no lock: it opens this Mac's GPU, which nothing it does changes.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
USAGE="usage: metal-compare.sh BASE [--head REV | --here DIR] [--out DIR]"

die() { # die <rc> <message>
  echo "metal-compare: $2" >&2
  exit "$1"
}

# shellcheck source=tools/nightly/units.sh
source "$ROOT/tools/nightly/units.sh" # archive_tree reads a commit, tree_commit labels this tree

FEATURES=pipeline-scan,encode-trace
# The trace opens the MLX split. An older base that cannot open it is named
# below; there is no GGUF fallback.
# PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
SPLIT=${BLOOMERY_FIXTURE_ROOT:-/models/fixtures}/qwen35moe-mlxsplit
shopt -s nullglob
mlx_shards=("$SPLIT"/*-00001-of-*.gguf)
shopt -u nullglob
if [ "${#mlx_shards[@]}" -ne 1 ]; then
  die 66 "qwen35moe-mlxsplit has ${#mlx_shards[@]} first shards under $SPLIT"
fi
FIXTURE=${mlx_shards[0]}
WORK=${METAL_COMPARE_WORK:-$HOME/.cache/bloomery/metal-compare}
TARGET=${METAL_COMPARE_TARGET:-$ROOT/target/metal-compare}
SHALLOW_IDS=760,6511,314 # the e2e smoke's prompt (lab-metal-e2e), depth ~0
DEEP_ROWS=600
DECODE=4 # decode tokens a trace covers

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die 78 "refused: the Metal move proof runs on an Apple Silicon Mac"

base= head_rev= here= out=
while [ $# -gt 0 ]; do
  case $1 in
    --head) [ $# -ge 2 ] || die 64 "$USAGE"; head_rev=$2; shift 2 ;;
    --here) [ $# -ge 2 ] || die 64 "$USAGE"; here=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || die 64 "$USAGE"; out=$2; shift 2 ;;
    -*) die 64 "$1 is no flag of this tool; $USAGE" ;;
    *) [ -z "$base" ] || die 64 "$USAGE"; base=$1; shift ;;
  esac
done
[ -n "$base" ] || die 64 "$USAGE"
[ -z "$head_rev" ] || [ -z "$here" ] || die 64 "--head and --here are one choice"
[ -f "$FIXTURE" ] || die 64 "the fixture $FIXTURE is not there (BLOOMERY_FIXTURE_ROOT)"

has_instrument() { # has_instrument <dir>: the tree carries the scan and the encode trace
  [ -f "$1/crates/metal/src/trace.rs" ] && [ -f "$1/crates/metal/src/scan.rs" ] &&
    grep -q '^encode-trace' "$1/crates/metal/Cargo.toml"
}

# archive_paths <rev> <dest> <path>: the tree of <path> at <rev> into <dest> (through a file, as archive_tree does)
archive_paths() {
  local f
  f=$(mktemp "${TMPDIR:-/tmp}/metal-compare-archive.XXXXXX") || die 70 "no temporary file"
  git -C "$ROOT" archive --format=tar -o "$f" "$1" "$3" && tar -x -f "$f" -C "$2"
  local rc=$?
  rm -f "$f"
  return "$rc"
}

# make_overlay <patch file>: the instrument as a patch against a tree that lacks it, from this tree's own history
make_overlay() {
  local patch=$1 t ic rc
  has_instrument "$ROOT" || die 2 "this tree has no instrument (crates/metal/src/trace.rs, scan.rs, the encode-trace feature): nothing to lay over a side"
  t=$(mktemp -d "${TMPDIR:-/tmp}/metal-compare-overlay.XXXXXX") || die 70 "no temporary directory"
  mkdir "$t/pre" "$t/post"
  ic=$(git -C "$ROOT" log --diff-filter=A --format=%H -n1 -- crates/metal/src/trace.rs)
  if [ -n "$ic" ]; then
    archive_paths "$ic^" "$t/pre" crates/metal && archive_paths "$ic" "$t/post" crates/metal || die 1 "cannot read the instrument commit ${ic:0:12}"
  else
    archive_paths HEAD "$t/pre" crates/metal || die 1 "cannot read HEAD's crates/metal"
    mkdir -p "$t/post/crates" && cp -R "$ROOT/crates/metal" "$t/post/crates/metal" || die 1 "cannot copy this tree's crates/metal"
  fi
  (cd "$t" && diff -ruN pre/crates/metal post/crates/metal) > "$patch"
  rc=$?
  rm -rf "$t"
  [ "$rc" = 1 ] || die 1 "the instrument's diff is empty or failed (diff rc $rc)"
}

# side_tree <name> <rev>: the directory of <rev>'s tree, extracted once and given the instrument when it has none
side_tree() {
  local name=$1 rev=$2 sha dir track want patch
  sha=$(git -C "$ROOT" rev-parse --verify --quiet "$rev^{commit}") || die 64 "'$rev' names no commit of this repository"
  track=$(basename "$ROOT")
  dir=$WORK/$track/${sha:0:12}
  mkdir -p "$WORK/$track" || die 70 "cannot make $WORK/$track"
  if [ -f "$dir/.metal-compare-ok" ]; then
    if has_instrument "$dir" && [ ! -f "$dir/.metal-compare-overlay" ]; then
      printf '%s' "$dir"
      return 0
    fi
  fi
  patch=$(mktemp "${TMPDIR:-/tmp}/metal-compare-overlay.XXXXXX") || die 70 "no temporary file"
  want=native
  if [ -f "$dir/.metal-compare-ok" ] && [ -f "$dir/.metal-compare-overlay" ]; then
    make_overlay "$patch"
    want=$(md5 -q "$patch")
    if [ "$(cat "$dir/.metal-compare-overlay")" = "$want" ]; then
      rm -f "$patch"
      printf '%s' "$dir"
      return 0
    fi
  fi
  case $dir in "$WORK"/*) rm -rf "$dir" ;; *) die 70 "refusing to clear $dir" ;; esac
  mkdir -p "$dir" || die 70 "cannot make $dir"
  echo "metal-compare: $name: extracting ${sha:0:12} into $dir" >&2
  archive_tree "$ROOT" "$sha" "$dir" || die 1 "git archive ${sha:0:12} failed (git $ARCHIVE_GIT_RC, tar $ARCHIVE_TAR_RC)"
  if ! has_instrument "$dir"; then
    [ -s "$patch" ] || make_overlay "$patch"
    echo "metal-compare: $name: ${sha:0:12} has no instrument, laying the overlay over it" >&2
    (cd "$dir" && patch -p1 -s < "$patch") || die 2 "$name: the instrument does not apply to ${sha:0:12} (patch failed): use a base that has crates/metal/src/trace.rs"
    [ -z "$(find "$dir" -name '*.rej' -o -name '*.orig' | head -1)" ] || die 2 "$name: the instrument applies to ${sha:0:12} only with rejects or fuzz: use a base that has it"
    md5 -q "$patch" | tr -d '\n' > "$dir/.metal-compare-overlay"
  fi
  rm -f "$patch"
  : > "$dir/.metal-compare-ok"
  printf '%s' "$dir"
}

# build_side <name> <dir> <out dir>: the side's scan and generate binaries, kept in <out dir>
build_side() {
  local name=$1 dir=$2 o=$3 rc t
  t=$TARGET/$(basename "$dir")-$(printf '%s' "$dir" | md5 -q | cut -c1-8)
  mkdir -p "$o" || die 70 "cannot make $o"
  echo "metal-compare: $name: building $dir ($FEATURES) in $t" >&2
  (cd "$dir" && CARGO_TARGET_DIR="$t" cargo build --release -p bloomery-metal --features "$FEATURES" --bin metal_scan --bin generate_metal) > "$o/build.log" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    tail -n 30 "$o/build.log" >&2
    die 1 "$name: the build ended rc $rc ($o/build.log)"
  fi
  cp "$t/release/metal_scan" "$t/release/generate_metal" "$o/" || die 1 "$name: cannot keep the binaries"
}

# scan_side <name> <out dir> <file>: the side's pipeline scan, into <out dir>/<file>
scan_side() {
  local name=$1 o=$2 f=$3
  "$o/metal_scan" --out "$o/$f" 2> "$o/scan.log" || { cat "$o/scan.log" >&2; die 1 "$name: metal_scan failed ($o/scan.log)"; }
}

# same_scan <scan binary> <scan> <scan>: two scans hold the same rows and hashes
same_scan() {
  "$1" --compare "$2" "$3" 2> /dev/null | grep -q '^summary: added=0 removed=0 changed=0 '
}

# trace_side <name> <out dir>: the side's encode traces and generated ids
trace_side() {
  local name=$1 o=$2 deep rc
  deep=$(awk -v n="$DEEP_ROWS" 'BEGIN { for (i = 1; i <= n; i++) printf "%s%d", (i > 1 ? "," : ""), (i * 7919) % 90001 + 1000 }')
  echo "metal-compare: $name: trace at depth ~0 and $DEEP_ROWS" >&2
  BLOOMERY_REF_MODEL=$FIXTURE "$o/generate_metal" --tokens "$SHALLOW_IDS" -n "$DECODE" --print-ids --encode-trace "$o/trace-d0.txt" > "$o/ids-d0.txt" 2> "$o/gen-d0.log"
  rc=$?
  [ "$rc" = 0 ] || {
    echo "metal-compare: $name did not run on the MLX split $FIXTURE (rc $rc); there is no GGUF fallback" >&2
    tail -n 20 "$o/gen-d0.log" >&2
    die 1 "$name: generate_metal (depth ~0) ended rc $rc ($o/gen-d0.log)"
  }
  BLOOMERY_REF_MODEL=$FIXTURE "$o/generate_metal" --tokens "$deep" -n "$DECODE" --print-ids --encode-trace "$o/trace-d$DEEP_ROWS.txt" > "$o/ids-d$DEEP_ROWS.txt" 2> "$o/gen-d$DEEP_ROWS.log"
  rc=$?
  [ "$rc" = 0 ] || {
    echo "metal-compare: $name did not run on the MLX split $FIXTURE (rc $rc); there is no GGUF fallback" >&2
    tail -n 20 "$o/gen-d$DEEP_ROWS.log" >&2
    die 1 "$name: generate_metal (depth $DEEP_ROWS) ended rc $rc ($o/gen-d$DEEP_ROWS.log)"
  }
}

# --- the sides
base_sha=$(git -C "$ROOT" rev-parse --verify --quiet "$base^{commit}") || die 64 "'$base' names no commit of this repository"
if [ -n "$here" ]; then
  [ -d "$here/crates/metal" ] || die 64 "--here $here: no crates/metal in it"
  has_instrument "$here" || die 2 "--here $here has no instrument; a directory is never patched"
  head_dir=$(cd "$here" && pwd)
  head_label=here-$(basename "$head_dir")
elif [ -n "$head_rev" ]; then
  head_sha=$(git -C "$ROOT" rev-parse --verify --quiet "$head_rev^{commit}") || die 64 "'$head_rev' names no commit of this repository"
  head_label=${head_sha:0:12}
  head_dir=$(side_tree head "$head_sha") || exit $?
else
  has_instrument "$ROOT" || die 2 "this tree has no instrument; give --head REV for a commit that has one"
  tree_commit "$ROOT" 12 || true
  head_label=$TREE_COMMIT
  head_dir=$ROOT
fi
base_dir=$(side_tree base "$base_sha") || exit $?
[ -n "$out" ] || out=$ROOT/target/metal-compare-out/${base_sha:0:12}-$head_label
rm -rf "$out/base" "$out/head"
mkdir -p "$out/base" "$out/head" || die 70 "cannot make $out"

echo "metal-compare: base ${base_sha:0:12} ($base_dir) against head $head_label ($head_dir)" >&2
build_side base "$base_dir" "$out/base"
build_side head "$head_dir" "$out/head"
# The scans, next to each other, each side twice in two processes: the device compiler's output for one kernel carries a few
# numbers that move with its session state (every row of a scan at once, no size changing), so a side counts only when a second
# process gives it the same scan, and the two sides' scans must not differ in every hash and nothing else (metal_scan --compare rc 7); a
# pair that fails either is scanned again, three times at most.
attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "metal-compare: scans, attempt $attempt" >&2
  scan_side base "$out/base" pipelines.tsv
  scan_side head "$out/head" pipelines.tsv
  scan_side base "$out/base" pipelines-2.tsv
  scan_side head "$out/head" pipelines-2.tsv
  if same_scan "$out/head/metal_scan" "$out/base/pipelines.tsv" "$out/base/pipelines-2.tsv" &&
    same_scan "$out/head/metal_scan" "$out/head/pipelines.tsv" "$out/head/pipelines-2.tsv"; then
    # the two sides in one state: a compare that moves every row's hash and nothing else (rc 7) is the state, not a kernel
    "$out/head/metal_scan" --compare "$out/base/pipelines.tsv" "$out/head/pipelines.tsv" > /dev/null 2>&1
    [ $? = 7 ] || break
    why="the two sides' scans differ in every hash and in nothing else"
  else
    why="a side did not reproduce its own scan in a second process"
  fi
  [ "$attempt" -lt 3 ] || die 1 "$why, in 3 attempts ($out/*/pipelines*.tsv): the compiler's session state keeps moving, nothing is proven (a change that moves every kernel and no size looks the same: read the archives with metal_scan --keep)"
  echo "metal-compare: $why (the compiler's session state moved): scanning again" >&2
done
trace_side base "$out/base"
trace_side head "$out/head"

# --- the compares: the head side's binary reads both (one format)
scan=$out/head/metal_scan
{
  echo "metal-compare: base ${base_sha:0:12} against head $head_label, fixture $(basename "$FIXTURE")"
  echo "scans: each side reproduced its own in a second process (attempt $attempt)"
  echo "== pipelines (metal_scan --compare)"
  "$scan" --compare "$out/base/pipelines.tsv" "$out/head/pipelines.tsv"
  echo "rc=$?"
  for d in 0 "$DEEP_ROWS"; do
    echo "== encode trace, depth $([ "$d" = 0 ] && echo '~0' || echo "$d") (metal_scan --compare-trace)"
    "$scan" --compare-trace "$out/base/trace-d$d.txt" "$out/head/trace-d$d.txt"
    echo "rc=$?"
    if cmp -s "$out/base/ids-d$d.txt" "$out/head/ids-d$d.txt"; then
      echo "generated ids: identical ($(cat "$out/head/ids-d$d.txt"))"
    else
      echo "generated ids: DIFFER"
      echo "  base: $(cat "$out/base/ids-d$d.txt")"
      echo "  head: $(cat "$out/head/ids-d$d.txt")"
    fi
  done
} > "$out/compare.txt" 2>&1
cat "$out/compare.txt"

rc_of() { grep -A0 -n '^rc=' "$out/compare.txt" | sed -n "$1s/.*rc=//p"; }
pipe_rc=$(rc_of 1)
t0_rc=$(rc_of 2)
t6_rc=$(rc_of 3)
ids_differ=$(grep -c '^generated ids: DIFFER' "$out/compare.txt")
added=$(sed -n 's/^summary: added=\([0-9]*\) .*/\1/p' "$out/compare.txt")
{ [ "$pipe_rc" = 2 ] || [ "$t0_rc" = 2 ] || [ "$t6_rc" = 2 ]; } && die 2 "a compare refused its input (see above)"
{ [ -n "$pipe_rc" ] && [ -n "$t0_rc" ] && [ -n "$t6_rc" ]; } || die 1 "cannot read the compares' exit codes from $out/compare.txt"
rc=0
case $pipe_rc in
  0) ;;
  3 | 4 | 5) rc=$pipe_rc ;;
  *) die 1 "metal_scan --compare ended rc $pipe_rc" ;;
esac
if [ "$t0_rc" != 0 ] || [ "$t6_rc" != 0 ] || [ "$ids_differ" != 0 ]; then
  case $rc in 0 | 3) rc=3 ;; *) ;; esac
fi
if [ "$rc" = 0 ] && [ "${added:-0}" != 0 ]; then rc=6; fi
echo "metal-compare: rc $rc (pipelines rc $pipe_rc, added ${added:-0}; trace depth ~0 rc $t0_rc, depth $DEEP_ROWS rc $t6_rc; ids differ $ids_differ) $out/compare.txt" >&2
exit "$rc"
