#!/usr/bin/env bash
# shellcheck shell=bash
# The reference trees of the oracles, checked and not trusted. Sourced; never executed. The one owner of that check: the
# EmbeddingGemma writers and the tokenizer oracle (crates/tokenizer/tools/oracle.sh) both go through it.
#
#   oracle_tree [--patch HASH] [--unversioned] [--built-by TEXT] WHO TREE PIN [BINARY...]
#   oracle_patch_hash TREE BASE
#
# A tree is an upstream commit, with a patch on it or without. PIN is that commit's full hash. Returns 0 and prints nothing, or
# exits 2 with a line that begins WHO and names what is wrong. It reads the tree and writes nothing into it: a script that
# wants the oracle calls this and never builds one.
#
# Without --patch the tree is the commit: its HEAD is PIN and it has no tracked change.
#
# With --patch the tree is the commit plus one patch: its HEAD descends from PIN, and its tracked files differ from PIN by
# the patch whose hash is HASH, local commits and uncommitted changes alike. `oracle_patch_hash TREE BASE` prints that hash
# (12 hex digits of the sha256 of `git diff BASE`, spelled one way whatever the git config), the value a profile pins and a
# set's `# build` line states after `<PIN's first 9 digits>+`. A rebuild of the same sources, or a clone that applies the same
# patch, has the same identity; another commit or another patch does not.
#
# Each BINARY (default llama-server) under TREE/build/bin must exist, and must be the tree's build:
#   - by its own report: `--version` names a commit that is a prefix of the tree's HEAD. A tree rebuilt without a readable git
#     reports `commit unknown` there and is refused: it is not the commit's build.
#   - with --unversioned (a binary that reports no commit, as ik's llama-tokenize), by its age: it and the libllama.so it
#     loads from TREE must be newer than every tracked file under src/, include/ and examples/tokenize/. A tree whose files
#     moved after the build is refused. The patch's effect is the caller's to probe where the patch has a visible one.
# --built-by TEXT is the remedy a missing file names in its error.
oracle_tree() {
  local patch='' versioned=1 built_by=''
  while [ $# -gt 0 ]; do
    case $1 in
      --patch) [ $# -ge 2 ] || { echo "oracle_tree: --patch takes a hash" >&2; exit 2; }; patch=$2; shift 2 ;;
      --unversioned) versioned=''; shift ;;
      --built-by) [ $# -ge 2 ] || { echo "oracle_tree: --built-by takes a text" >&2; exit 2; }; built_by=" ($2)"; shift 2 ;;
      *) break ;;
    esac
  done
  [ $# -ge 3 ] || { echo "oracle_tree: want [--patch HASH] [--unversioned] [--built-by TEXT] WHO TREE PIN [BINARY...]" >&2; exit 2; }
  local who=$1 tree=$2 pin=$3 head have bin report rc got file lib
  local re='commit ([0-9a-f]{7,40})([^0-9a-f]|$)'
  shift 3
  [ -d "$tree" ] || { echo "$who: no oracle tree at $tree$built_by" >&2; exit 2; }
  head=$(git -c safe.directory="$tree" -C "$tree" rev-parse HEAD) || { echo "$who: $tree is not a git tree" >&2; exit 2; }
  if [ -z "$patch" ]; then
    [ "$head" = "$pin" ] || { echo "$who: $tree is at $head, the profile pins $pin" >&2; exit 2; }
    have=$(git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no) ||
      { echo "$who: cannot inspect tracked changes in $tree" >&2; exit 2; }
    [ -z "$have" ] || { echo "$who: $tree has tracked changes; the oracle is the commit's build" >&2; exit 2; }
  else
    # rc 1 is "not an ancestor"; any other failure is a pin the tree does not hold (128).
    rc=0
    git -c safe.directory="$tree" -C "$tree" merge-base --is-ancestor "$pin" "$head" 2>/dev/null || rc=$?
    [ "$rc" != 128 ] || { echo "$who: $tree does not hold the pinned commit $pin" >&2; exit 2; }
    [ "$rc" = 0 ] || { echo "$who: $tree is at $head, which does not descend from the pinned commit $pin" >&2; exit 2; }
    got=$(oracle_patch_hash "$tree" "$pin") || { echo "$who: cannot diff $tree against $pin" >&2; exit 2; }
    [ "$got" = "$patch" ] ||
      { echo "$who: $tree's tracked files differ from ${pin:0:9} by patch $got, the profile pins patch $patch" >&2; exit 2; }
  fi
  [ $# -gt 0 ] || set -- llama-server
  for bin in "$@"; do
    [ -x "$tree/build/bin/$bin" ] || { echo "$who: no $tree/build/bin/$bin$built_by" >&2; exit 2; }
    if [ -n "$versioned" ]; then
      report=$("$tree/build/bin/$bin" --version 2>&1) || { echo "$who: $tree/build/bin/$bin --version failed: $report" >&2; exit 2; }
      # The whole commit token, 7 hex digits or more, a prefix of the tree's HEAD.
      [[ $report =~ $re ]] && [ "${head:0:${#BASH_REMATCH[1]}}" = "${BASH_REMATCH[1]}" ] ||
        { echo "$who: $tree/build/bin/$bin is not built at ${head:0:7}: $report" >&2; exit 2; }
    else
      command -v ldd >/dev/null || { echo "$who: no ldd to find the libllama.so $tree/build/bin/$bin loads" >&2; exit 2; }
      lib=$(ldd "$tree/build/bin/$bin" 2>/dev/null | awk -v t="$tree/" 'index($3, t) == 1 && $1 ~ /^libllama\.so/ { print $3; exit }')
      [ -n "$lib" ] || { echo "$who: $tree/build/bin/$bin loads no libllama.so from $tree" >&2; exit 2; }
      for got in "$tree/build/bin/$bin" "$lib"; do
        file=$(oracle_newer_source "$tree" "$got") || { echo "$who: cannot list the tracked sources of $tree" >&2; exit 2; }
        [ -z "$file" ] ||
          { echo "$who: $got is older than $tree/$file, a tracked file: it is not this tree's build" >&2; exit 2; }
      done
    fi
  done
}

# oracle_patch_hash TREE BASE: the 12 hex digits that name TREE's tracked files against the commit BASE. The diff is spelled
# with every option that could change its bytes pinned, so any git on any machine prints the same for the same files.
oracle_patch_hash() {
  local tree=$1 base=$2 out sum
  out=$(mktemp) || return 1
  git -c safe.directory="$tree" -c core.quotepath=off -C "$tree" diff --no-ext-diff --no-textconv --no-color --no-renames \
    --binary --full-index --diff-algorithm=myers --src-prefix=a/ --dst-prefix=b/ -U3 "$base" -- > "$out" || { rm -f "$out"; return 1; }
  if command -v sha256sum >/dev/null; then sum=$(sha256sum < "$out"); else sum=$(shasum -a 256 < "$out"); fi
  rm -f "$out"
  printf '%s\n' "${sum:0:12}"
}

# oracle_newer_source TREE FILE: the first tracked file of TREE under src/, include/ or examples/tokenize/ that is newer than
# FILE, or nothing. A listing that fails is a failure, never an empty answer.
oracle_newer_source() {
  local tree=$1 ref=$2 list newer
  list=$(mktemp) || return 1
  git -c safe.directory="$tree" -C "$tree" ls-files -z -- src include examples/tokenize > "$list" || { rm -f "$list"; return 1; }
  newer=$(cd "$tree" && xargs -0 -r sh -c 'find "$@" -maxdepth 0 -newer "$0" -print' "$ref" < "$list") || { rm -f "$list"; return 1; }
  rm -f "$list"
  printf '%s\n' "$newer" | sed -n 1p
}
